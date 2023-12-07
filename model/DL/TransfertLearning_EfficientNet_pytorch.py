import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import timm

import matplotlib.pyplot as plt
from tqdm import tqdm

# Parameters
batch_size = 32
image_size = (366, 366)

# Data augmentation and normalization
transform = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ]
)

train_folder = "data\PBC_APL_train"
test_folder = "data\PBC_APL_test"
val_folder = "data\PBC_APL_val"

# Define the dataset for each split (train, test, val)
train_dataset = ImageFolder(root=train_folder, transform=transform)
test_dataset = ImageFolder(root=test_folder, transform=transform)
val_dataset = ImageFolder(root=val_folder, transform=transform)

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False):
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_accuracy_max = 0.0

    def __call__(self, val_accuracy, model, optimizer, epoch):
        score = val_accuracy

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_accuracy, model, optimizer, epoch)
            self.counter = 0

    def save_checkpoint(self, val_accuracy, model, optimizer, epoch):
        if self.verbose:
            print(
                f"Validation accuracy increased ({self.val_accuracy_max:.6f} --> {val_accuracy:.6f}).  Saving model ..."
            )
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "accuracy": val_accuracy,
            },
            "Models/checkpoint_transfert_learning_efficientnet_b1_v4_fine_tuned.pth",
        )
        self.val_accuracy_max = val_accuracy


# Define the model class
class CustomEfficientNetV2(nn.Module):
    def __init__(self):
        super(CustomEfficientNetV2, self).__init__()
        self.effnet = timm.create_model(
            "tf_efficientnetv2_b1", pretrained=True, num_classes=9
        )
        self.gradients = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, x):
        x = self.effnet(x)
        return x

    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self, x):
        h = x.register_hook(self.activations_hook)
        x = self.forward(x)
        return x

    def remove_hook(self):
        if self.hook is not None:
            self.hook.remove()


# Instantiate the CustomEfficientNetV2 model
model = CustomEfficientNetV2()

# Move the model to the device
model = model.to(device)

# Specify the number of blocks to unfreeze
num_blocks_to_unfreeze = 1

# Get the total number of blocks in the EfficientNetV2 model
total_blocks = len(model.effnet.blocks)

# Calculate the starting block index to unfreeze
start_block_index = max(0, total_blocks - num_blocks_to_unfreeze)

# Unfreeze the desired blocks
for i, block in enumerate(model.effnet.blocks):
    if i >= start_block_index:
        for param in block.parameters():
            param.requires_grad = True
    else:
        for param in block.parameters():
            param.requires_grad = False


# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

# Instantiate EarlyStopping
early_stopping = EarlyStopping(patience=5, delta=0, verbose=True)

# Lists to store training and validation accuracies
train_accuracies = []
val_accuracies = []

# Check if a checkpoint file exists
checkpoint_path = (
    "Models/checkpoint_transfert_learning_efficientnet_b1_v4_fine_tuned.pth"
)
if os.path.exists(checkpoint_path):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])

    # Load optimizer state
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load other training-related parameters
    start_epoch = checkpoint["epoch"] + 1
    early_stopping.val_accuracy_max = checkpoint["accuracy"]

    print(f"Checkpoint loaded. Resuming training from epoch {start_epoch}.")

else:
    start_epoch = 0
    print("No checkpoint found. Starting training from epoch 1.")

# Training loop with tqdm
num_epochs = 20

for epoch in range(start_epoch, start_epoch + num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    tqdm_train_loader = tqdm(
        train_loader, desc=f"Epoch {epoch + 1}/{start_epoch + num_epochs}"
    )

    for inputs, labels in tqdm_train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Calculate training accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

        tqdm_train_loader.set_postfix(loss=running_loss / len(tqdm_train_loader))

    epoch_loss = running_loss / len(train_loader)
    train_accuracy = correct_train / total_train

    # Validation step
    model.eval()
    with torch.no_grad():
        correct_val = 0
        total_val = 0
        for inputs_val, labels_val in val_loader:
            inputs_val, labels_val = inputs_val.to(device), labels_val.to(device)
            outputs_val = model(inputs_val)
            _, predicted_val = torch.max(outputs_val.data, 1)
            total_val += labels_val.size(0)
            correct_val += (predicted_val == labels_val).sum().item()

        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Call the EarlyStopping callback
        early_stopping(val_accuracy, model, optimizer, epoch)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(
        f"Epoch {epoch + 1}/{start_epoch + num_epochs}, Loss: {epoch_loss:.4f}, Training Accuracy: {train_accuracy:.4f}, Validation Accuracy: {val_accuracy:.4f}"
    )

    # Store training accuracy
    model.train()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    train_accuracy = correct / total
    train_accuracies.append(train_accuracy)

# Save the final model
torch.save(
    model.state_dict(), "Models/efficientnetv2_transfer_learning_b1_v4_fine_tuned.pth"
)

# Plotting the accuracies
plt.plot(train_accuracies, label="Training Accuracy")
plt.plot(val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("Models/Accuracy_fig/Accuracy_efficientnet_b1_v4.png")
plt.show()
