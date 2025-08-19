# Skin Cancer Analysis

## Project Overview
This project focuses on the analysis and classification of skin cancer using deep learning techniques. The goal is to accurately identify different types of skin cancer from dermatoscopic images, helping in early detection and diagnosis. The trained model achieved an **Weighted F1 Score of 85%** on the test set, demonstrating strong performance in classifying skin cancer types.

Demo app link : https://huggingface.co/spaces/mdfazil-08/skin-cancer-analysis

## Types of Skin Cancer
The HAM10000 dataset includes several types of skin cancer lesions, such as:
- **Melanoma**: A dangerous form of skin cancer that can spread rapidly.
- **Melanocytic nevi**: Common moles, usually benign.
- **Basal cell carcinoma**: The most common type of skin cancer, typically slow-growing.
- **Actinic keratoses**: Precancerous lesions caused by sun damage.
- **Benign keratosis-like lesions**: Non-cancerous growths.
- **Dermatofibroma**: Benign skin growths.
- **Vascular lesions**: Lesions related to blood vessels.

## How the Project Was Made
The workflow for this project is as follows:

1. **Download the HAM10000 Dataset**
   - Images and metadata were downloaded from [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
   - The metadata.csv file was used to extract labels.csv for classification purposes.

2. **Data Splitting**
   - The dataset was split into training and testing sets.
   - Splitting was performed based on the type of skin cancer to ensure balanced representation.

3. **Model Training**
   - A ResNet50 architecture was used for image classification.
   - The model was trained for 50 epochs to achieve optimal performance.

4. **Streamlit Application**
   - A Streamlit app was developed to allow users to upload and analyze skin images interactively.

## Dataset Source
- [HAM10000 Skin Cancer Dataset on Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

## Getting Started
To run the project, follow these steps:
1. Install dependencies (see requirements.txt or environment.yml).
2. Download the dataset from Kaggle and place it in the dataset/ folder.
3. Run the training script to train the model.
4. Launch the Streamlit app to analyze images.

---
This project aims to support research and development in skin cancer detection using AI and deep learning.
