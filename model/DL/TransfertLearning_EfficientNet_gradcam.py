import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt

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
        if self.hook is not None:
            self.hook.remove()


# Initialize the model
model = CustomEfficientNetV2().to(device)
model_path = "Models/efficientnetv2_transfer_learning_b1_v4.pth"
model.load_state_dict(torch.load(model_path))
model.to(device).eval()

# Load the image from the dataloader
dataset = datasets.ImageFolder(root="data/Test", transform=transform)
dataloader = data.DataLoader(dataset=dataset, shuffle=False, batch_size=1)
img, _ = next(iter(dataloader))
img = img.to(device)


# GradCAM Class
class GradCam:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.feature_maps = None
        self.gradients = None
        self.hook = self.register_hooks()

    def register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = dict(self.model.named_modules())[self.target_layer_name]
        forward_handle = target_layer.register_forward_hook(forward_hook)
        backward_handle = target_layer.register_full_backward_hook(backward_hook)

        return forward_handle, backward_handle

    def generate_heatmap(self, input_image, target_class):
        self.model.zero_grad()
        output = self.model(input_image)
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.feature_maps, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        return heatmap

    def remove_hooks(self):
        for handle in self.hook:
            handle.remove()


target_layer_name = "effnet.conv_head"

class_names = {
    0: "Basophil",
    1: "Blast, no lineage spec",
    2: "Eosinophil",
    3: "Erythroblast",
    4: "Ig",
    5: "Lymphocyte",
    6: "Monocyte",
    7: "Neutrophil",
    8: "Platelet",
}


# Function to generate GradCAM and display it along with the original image
def generate_and_display_gradcam(model, dataloader, target_layer_name, image_size):
    # Instantiate GradCam
    grad_cam = GradCam(model, target_layer_name)

    # Get one image from each class in the test dataloader
    class_images = {}
    for imgs, labels in dataloader:
        for img, label in zip(imgs, labels):
            img = img.to(device)
            label = label.item()

            # Unsqueeze to add batch dimension
            img = img.unsqueeze(0)

            # Get predicted class
            with torch.no_grad():
                prediction = model(img)
            predicted_class = torch.argmax(prediction).item()

            # Generate heatmap
            heatmap = grad_cam.generate_heatmap(img, target_class=predicted_class)

            # Convert heatmap to numpy array
            heatmap_np = heatmap.cpu().squeeze().detach().numpy()

            # Resize heatmap to the original image size
            heatmap_resized = cv2.resize(heatmap_np, (image_size[1], image_size[0]))

            # Normalize and colorize the heatmap
            heatmap_normalized = (heatmap_resized - np.min(heatmap_resized)) / (
                np.max(heatmap_resized) - np.min(heatmap_resized) + 1e-8
            )
            heatmap_colored = plt.get_cmap("jet")(heatmap_normalized)

            # Apply heatmap overlay on the original image
            original_image = img.squeeze().cpu().numpy().transpose((1, 2, 0))
            alpha = 0.5
            result = alpha * heatmap_colored[:, :, :3] + (1 - alpha) * original_image

            # Store the results for each class
            class_images[label] = {
                "original_image": original_image,
                "heatmap_overlay": result,
                "predicted_class": predicted_class,
            }

            # Break out of the inner loop after processing the first image for each class
            break

    # Display the images in a 3x3 subplot
    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    for i in range(3):
        for j in range(3):
            class_label = i * 3 + j
            if class_label in class_images:
                axs[i, j].imshow(class_images[class_label]["heatmap_overlay"])
                axs[i, j].set_title(
                    f"Predicted: {class_names[class_images[class_label]['predicted_class']]}\nTrue Label: {class_names[class_label]}",
                    fontsize=20,
                )
                axs[i, j].axis("off")

    plt.tight_layout()
    plt.savefig("Models/Gradcam/gradcam_subplot.png")
    plt.show()


# Call the function with the test dataloader
generate_and_display_gradcam(model, test_loader, target_layer_name, image_size)
