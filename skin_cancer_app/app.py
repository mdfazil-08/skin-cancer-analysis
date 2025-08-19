import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Skin cancer class labels
class_names = [
    "Actinic Keratoses and Intraepithelial Carcinoma",
    "Basal Cell Carcinoma",
    "Benign Keratosis-like Lesions",
    "Dermatofibroma",
    "Melanoma",
    "Melanocytic Nevi",
    "Vascular Lesions"
]

# Description for each class
class_descriptions = {
    "Actinic Keratoses and Intraepithelial Carcinoma": "Pre-cancerous skin lesions that can develop into squamous cell carcinoma. ⚠️ Cancerous (pre-malignant).",
    "Basal Cell Carcinoma": "The most common type of skin cancer, slow-growing and rarely spreads. ✅ Cancerous.",
    "Benign Keratosis-like Lesions": "Non-cancerous skin growths that resemble warts or moles. ✅ Not cancerous.",
    "Dermatofibroma": "A harmless, firm bump under the skin, often from insect bites. ✅ Not cancerous.",
    "Melanoma": "A serious and aggressive form of skin cancer that can spread rapidly. ❗ Highly cancerous.",
    "Melanocytic Nevi": "Common moles that are generally benign but can evolve. ✅ Not cancerous, but monitor.",
    "Vascular Lesions": "Abnormal blood vessels often appearing as red/purple spots. ✅ Not cancerous."
}

# Path to saved model
MODEL_PATH = '/home/fazil/Projects/cv-projects/skin_cancer_app/model_epoch_50.pt'

# Image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model once using cache
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    state_dict = checkpoint['model_state_dict']
    # Remove 'module.' if present
    updated_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(updated_state_dict, strict=True)
    model.eval()
    return model

model = load_model()

# App UI
st.title("Skin Cancer Classification")
st.subheader("This app classifies skin lesions into different types of skin cancer.")
st.write("This app uses a pre-trained ResNet model to classify skin lesions into one of the following categories:")
for class_name in class_names:
    st.write(f"- {class_name}")
st.write('Disclaimer: This app is for educational purposes only and should not be used for medical diagnosis.')

st.write("Upload a skin lesion image to predict the type of skin cancer.")
uploaded_file = st.file_uploader("📷 Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # Preprocess image
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1).squeeze()
        confidence, pred_class = torch.max(probs, dim=0)

    predicted_label = class_names[pred_class]

    st.markdown(f"### 🩺 Prediction: `{predicted_label}`")
    st.markdown(f"**Confidence:** {confidence.item() * 100:.2f}%")
    st.info(class_descriptions[predicted_label])

    # Probability chart
    st.subheader("🔍 All Class Probabilities")
    fig, ax = plt.subplots()
    ax.barh(class_names, probs.numpy())
    ax.set_xlabel("Probability")
    ax.set_xlim([0, 1])
    st.pyplot(fig)
