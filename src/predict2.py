import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib


DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE   = 224      # EfficientNet expects 224×224
BATCH_SIZE = 32       # smaller than before — 224px images are heavier
EPOCHS     = 10
LR_HEAD    = 1e-3     # classifier head (randomly initialised)
LR_BODY    = 1e-4     # pretrained backbone (fine-tuning, slower)
MODEL_NAME = "meow"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

test_tf = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ── 5. Single-image inference ─────────────────────────────────
def predict_emotion(img_path: str, model, le) -> str:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    tensor = test_tf(img).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        idx = model(tensor).argmax(1).item()
    return le.inverse_transform([idx])[0]

def build_model(num_classes: int) -> nn.Module:
    """
    EfficientNet-B0 pretrained on ImageNet.
    We replace the final classifier layer with one sized for our emotions.
    The backbone is kept frozen for the first few epochs, then unfrozen.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze all backbone parameters initially
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier head (1280 → num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    # Head params are trainable from the start
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model

# ── 6. Load saved model (in a separate script) ────────────────
def load_model(model_name: str = MODEL_NAME):
    _le    = joblib.load(f"models/emotion_efficientnet_labels_{model_name}.pkl")
    _model = build_model(len(_le.classes_)).to(DEVICE)
    _model.load_state_dict(
        torch.load(f"models/emotion_efficientnet_{model_name}_best.pt",
                   map_location=DEVICE)
    )
    _model.eval()
    return _model, _le

m, enc = load_model()
print(predict_emotion("some_face.jpg", model=m, le=enc))