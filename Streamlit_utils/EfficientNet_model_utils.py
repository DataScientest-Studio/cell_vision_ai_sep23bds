# model.py

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import os


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


class BloodCellClassifier:
    def __init__(self, model_path):
        self.model = CustomEfficientNetV2()
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((366, 366)),
                transforms.ToTensor(),
            ]
        )

    def predict(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)
        with torch.no_grad():
            output = self.model(image)
            _, predicted_class = torch.max(output, 1)
        return predicted_class.item()

    def get_class_name(self, class_id):
        class_labels = {
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
        return class_labels[class_id]
