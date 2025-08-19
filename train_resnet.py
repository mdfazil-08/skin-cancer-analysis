import os
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report, confusion_matrix,
    precision_score, recall_score, f1_score, accuracy_score
)
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


DATA_DIR = '/home/fazil/Projects/cv-projects/dataset'
CHECKPOINT_DIR = '/home/fazil/Projects/cv-projects/checkpoints'
NUM_CLASSES = 7
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(DEVICE)}")


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


datasets_test = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=transform)
dataloader_test = DataLoader(datasets_test, batch_size=BATCH_SIZE, shuffle=False)
class_names = datasets_test.classes


def load_model_from_checkpoint():
    files = sorted([f for f in os.listdir(CHECKPOINT_DIR) if f.endswith('.pt')])
    if not files:
        raise FileNotFoundError("❌ No checkpoint found in CHECKPOINT_DIR")

    latest = files[-1]
    print(f"Loading model from checkpoint: {latest}")

    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, latest), map_location=DEVICE)

    model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model


def inference(model):
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in dataloader_test:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())


    print("\nClassification Report (per-class precision/recall/F1):")
    print(classification_report(all_labels, all_preds, target_names=class_names))

 
    cm = confusion_matrix(all_labels, all_preds)
    TP = np.diag(cm)
    FP = cm.sum(axis=0) - TP
    FN = cm.sum(axis=1) - TP
    TN = cm.sum() - (FP + FN + TP)

    print("\nMetrics (per class):")
    for i, name in enumerate(class_names):
        print(f"\nClass: {name}")
        print(f"  TP: {TP[i]}, FP: {FP[i]}, TN: {TN[i]}, FN: {FN[i]}")
        precision = TP[i] / (TP[i] + FP[i]) if TP[i] + FP[i] > 0 else 0
        recall    = TP[i] / (TP[i] + FN[i]) if TP[i] + FN[i] > 0 else 0
        print(f"  Precision: {precision:.2f}, Recall: {recall:.2f}")

    # Overall Metrics

    accuracy = accuracy_score(all_labels, all_preds)

    # Macro metrics → all classes treated equally
    macro_precision = precision_score(all_labels, all_preds, average="macro")
    macro_recall    = recall_score(all_labels, all_preds, average="macro")
    macro_f1        = f1_score(all_labels, all_preds, average="macro")

    # Weighted metrics → accounts for class imbalance
    weighted_precision = precision_score(all_labels, all_preds, average="weighted")
    weighted_recall    = recall_score(all_labels, all_preds, average="weighted")
    weighted_f1        = f1_score(all_labels, all_preds, average="weighted")

    print("\n===== Overall Evaluation =====")
    print(f"Accuracy: {accuracy:.4f}")

    print("\n--- Macro (treats all classes equally) ---")
    print(f"Precision (Macro): {macro_precision:.4f}")
    print(f"Recall (Macro):    {macro_recall:.4f}")
    print(f"F1-score (Macro):  {macro_f1:.4f}")

    print("\n--- Weighted (accounts for imbalance) ---")
    print(f"Precision (Weighted): {weighted_precision:.4f}")
    print(f"Recall (Weighted):    {weighted_recall:.4f}")
    print(f"F1-score (Weighted):  {weighted_f1:.4f}")


if __name__ == '__main__':
    model = load_model_from_checkpoint()
    inference(model)
