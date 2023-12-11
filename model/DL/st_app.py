import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import matplotlib.pyplot as plt

from EfficientNet_model_utils import BloodCellClassifier
from EfficientNet_gradcam_utils import generate_and_display_gradcam
from torchvision import transforms
from PIL import Image

st.title("Prédiction d'image selon le modèle")
model2 = tf.keras.models.load_model("mobilenetv2.keras")
model_path = "efficientnetv2_transfer_learning_b1_v4_fine_tuned.pth"
classifier = BloodCellClassifier(model_path)
model = tf.keras.models.load_model("model-21-0.89.h5")


# Function to prepare an image for Grad-CAM with CNN
def prepare_image(img, target_size=(256, 256)):
    # Préparer l'image pour l'entrée du modèle
    img = image.load_img(img, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array


# Function to apply Guided Grad-CAM on a specific image
def apply_guided_grad_cam(model, img_array, layer_name, target_size):
    # Create the gradient model
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).output, model.output]
    )

    # Use GradientTape to compute gradients
    with tf.GradientTape() as tape:
        inputs = tf.cast(img_array, tf.float32)
        conv_outputs, predictions = grad_model(inputs)
        loss = predictions[:, 0]

    # Compute gradients
    grads = tape.gradient(loss, conv_outputs)

    # Compute guided gradients
    cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
    cast_grads = tf.cast(grads > 0, "float32")
    guided_grads = cast_conv_outputs * cast_grads * grads

    # Remove batch dimension
    conv_outputs = conv_outputs[0]
    guided_grads = guided_grads[0]

    # Compute weights
    weights = tf.reduce_mean(guided_grads, axis=(0, 1))

    # Compute CAM
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)

    # Grab spatial dimensions of the input image and resize the CAM
    (w, h) = (img_array.shape[2], img_array.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))

    # Normalize the heatmap
    heatmap = np.maximum(heatmap, 0)
    heatmap /= heatmap.max()

    return heatmap


# Liste déroulante pour choisir le modèle
selected_model = st.selectbox(
    "Choisissez un modèle", ["CNN 'from scratch'", "MobileNetV2", "EfficientNetV2"]
)

# Téléchargement de l'image depuis l'utilisateur
uploaded_file = st.file_uploader("Téléchargez une image", type=["jpg", "png", "jpeg"])
# Charger l'image
if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))
    col1, col2 = st.columns(2)
    # Afficher l'image originale
    col1.image(img, caption="Image originale", width=224)


if selected_model == "CNN 'from scratch'":
    guided_grad_cam_layer = (
        "conv2d_8"  # You may need to adjust this based on your model architecture
    )
    if uploaded_file is not None:
        img = image.load_img(uploaded_file, target_size=(256, 256))
        img_array = prepare_image(uploaded_file)
        predictions = model.predict(img_array)
        # Obtenir l'indice de la classe prédite
        predicted_class_index = np.argmax(predictions)

        class_indices = {
            "basophil": 0,
            "blast, no lineage spec": 1,
            "eosinophil": 2,
            "erythroblast": 3,
            "ig": 4,
            "lymphocyte": 8,
            "monocyte": 5,
            "neutrophil": 6,
            "platelet": 7,
        }

        # Obtenir le nom de la classe prédite
        predicted_class_name = list(class_indices.keys())[
            list(class_indices.values()).index(predicted_class_index)
        ]

        # Afficher la classe prédite
        st.subheader(f"Classe prédite : {predicted_class_name}")
        # Afficher la Guided Grad-CAM
        guided_grad_cam_result = apply_guided_grad_cam(
            model, img_array, guided_grad_cam_layer, (256, 256)
        )
        heatmap = cv2.resize(
            guided_grad_cam_result, (img_array.shape[2], img_array.shape[1])
        )
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimposer la heatmap sur l'image originale
        superimposed_img = cv2.addWeighted(
            np.asarray(img).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0
        )
        col2.image(superimposed_img, caption="Grad-CAM", width=224)


elif selected_model == "MobileNetV2":
    # Choose the layer for Guided Grad-CAM
    guided_grad_cam_layer = (
        "out_relu"  # You may need to adjust this based on your model architecture
    )

    if uploaded_file is not None:
        # Convertir l'image en tableau numpy
        img_array = image.img_to_array(img)

        # Ajouter une dimension supplémentaire (batch_size)
        img_array = np.expand_dims(img_array, axis=0)

        # Prétraiter l'image pour la faire correspondre au format que le modèle attend
        img_array = preprocess_input(img_array)

        # Faire une prédiction
        predictions = model2.predict(img_array)

        # Obtenir l'indice de la classe prédite
        predicted_class_index = np.argmax(predictions)

        class_indices = {
            "basophil": 0,
            "blast, no lineage spec": 1,
            "eosinophil": 2,
            "erythroblast": 3,
            "ig": 4,
            "lymphocyte": 5,
            "monocyte": 6,
            "neutrophil": 7,
            "platelet": 8,
        }

        # Obtenir le nom de la classe prédite
        predicted_class_name = list(class_indices.keys())[
            list(class_indices.values()).index(predicted_class_index)
        ]

        # Afficher la classe prédite
        st.subheader(f"Classe prédite : {predicted_class_name}")
        # Afficher la Guided Grad-CAM
        guided_grad_cam_result = apply_guided_grad_cam(
            model2, img_array, guided_grad_cam_layer, (224, 224)
        )
        heatmap = cv2.resize(
            guided_grad_cam_result, (img_array.shape[2], img_array.shape[1])
        )
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Superimposer la heatmap sur l'image originale
        superimposed_img = cv2.addWeighted(
            np.asarray(img).astype(np.uint8), 0.5, heatmap_rgb, 0.5, 0
        )
        col2.image(superimposed_img, caption="Grad-CAM", width=224)

elif selected_model == "EfficientNetV2":
    if uploaded_file is not None:
        # Load the classifier and make prediction
        prediction = classifier.predict(uploaded_file)

        # Display the prediction
        class_labels = {
            0: "basophil",
            1: "blast, no lineage spec",
            2: "eosinophil",
            3: "erythroblast",
            4: "ig",
            5: "lymphocyte",
            6: "monocyte",
            7: "neutrophil",
            8: "platelet",
        }

        predicted_class_name = class_labels[prediction]
        st.subheader(f"Classe prédite : {predicted_class_name}")

        ##### Fin de la partie prédiction #####

        # Convert image to torch.Tensor
        image = Image.open(uploaded_file).convert("RGB")
        transform = transforms.Compose(
            [
                transforms.Resize((366, 366)),
                transforms.ToTensor(),
            ]
        )
        input_image = transform(image).unsqueeze(0)
        image_size = (366, 366)

        # Get GradCAM
        target_layer_name = "effnet.conv_head"
        image_with_gradcam = generate_and_display_gradcam(
            classifier.model, input_image, target_layer_name, image_size
        )
        col2.image(image_with_gradcam, caption="Grad-CAM", width=224)
