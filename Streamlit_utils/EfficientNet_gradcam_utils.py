# gradcam_utils.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import timm
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


def generate_and_display_gradcam(model, input_image, target_layer_name, image_size):
    # Instantiate GradCam
    grad_cam = GradCam(model, target_layer_name)

    # Get predicted class
    with torch.no_grad():
        prediction = model(input_image)
    predicted_class = torch.argmax(prediction).item()

    # Generate heatmap
    heatmap = grad_cam.generate_heatmap(input_image, target_class=predicted_class)

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
    original_image = input_image.squeeze().cpu().numpy().transpose((1, 2, 0))
    alpha = 0.5
    result = alpha * heatmap_colored[:, :, :3] + (1 - alpha) * original_image

    # Display the images in a subplot
    fig = plt.figure(figsize=(10, 5))

    plt.imshow(result)
    plt.axis("off")

    return fig
