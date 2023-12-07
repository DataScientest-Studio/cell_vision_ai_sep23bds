import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

import timm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

train_folder = "data/PBC_APL_train"
test_folder = "data/PBC_APL_test"
val_folder = "data/PBC_APL_val"

# Define the dataset for each split (train, test, val)
train_dataset = ImageFolder(root=train_folder, transform=transform)
test_dataset = ImageFolder(root=test_folder, transform=transform)
val_dataset = ImageFolder(root=val_folder, transform=transform)

# Create DataLoaders for each split
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


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
        if hasattr(self, "hook") and self.hook is not None:
            self.hook.remove()


# Instantiate the CustomEfficientNetV2 model
model = CustomEfficientNetV2().to(device)

# Load the model
model_path = "Models/efficientnetv2_transfer_learning_b1_v4_fine_tuned.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Lists to store true labels and predicted labels
true_labels = []
predicted_labels = []

# Make predictions on the test set with tqdm
with torch.no_grad(), tqdm(total=len(test_loader), unit="batch") as pbar:
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

        pbar.update(1)  # Update the progress bar

# Convert lists to numpy arrays
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Get the class labels from the original ImageFolder dataset
class_labels = train_dataset.classes

# Print the classification report
report = classification_report(true_labels, predicted_labels, target_names=class_labels)
print(report)

# Save the classification report to a file
report_path = "Models/Classification_report/EfficientNet_b1_v4_fine_tuned.txt"
with open(report_path, "w") as f:
    f.write(report)
