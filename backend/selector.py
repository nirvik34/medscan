# %%
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# %%
import torch
import torch.nn as nn

# Correct architecture (must match training)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # two conv blocks with 2x pooling -> spatial reduction by 4
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(),
            # For input 128x128 -> after two pools -> 32x32 spatial -> 32*32*32 features
            nn.Linear(32*32*32, 128), nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# Load CT
checkpoint = torch.load("models/ct_model.pth", map_location="cpu")
ct_model = SimpleCNN(num_classes=checkpoint.get("num_classes", 2))
ct_model.load_state_dict(checkpoint["model_state_dict"])
ct_model.eval()

# Load X-ray
checkpoint = torch.load("models/cnn_chestxray.pth", map_location="cpu")
xray_model = SimpleCNN(num_classes=checkpoint.get("num_classes", 2))
xray_model.load_state_dict(checkpoint["model_state_dict"])
xray_model.eval()

# Load Ultrasound
checkpoint = torch.load("models/ultrasound_model.pth", map_location="cpu")
ultrasound_model = SimpleCNN(num_classes=checkpoint.get("num_classes", 2))
ultrasound_model.load_state_dict(checkpoint["model_state_dict"])
ultrasound_model.eval()

print("✅ All models loaded and ready for inference")


# %%
# Step 2: Image transforms
# ------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # RGB mean/std
])

# %%
def select_model(modality):
    if modality.lower() == "ct":
        return ct_model
    elif modality.lower() == "xray":
        return xray_model
    elif modality.lower() == "ultrasound":
        return ultrasound_model
    else:
        raise ValueError("Unknown modality. Please specify CT, X-ray, or Ultrasound.")

def predict(image_path, modality):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # add batch dimension

    if modality.lower() == "ct":
        model = ct_model
    elif modality.lower() == "xray":
        model = xray_model
    elif modality.lower() == "ultrasound":
        model = ultrasound_model
    else:
        raise ValueError("Invalid modality! Choose from: ct, xray, ultrasound")

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred = torch.max(probs, 1)

    return pred.item(), confidence.item()

# %%
modality = input("Enter modality (ct / xray / ultrasound): ")
image_path = input("Enter image path: ")

# %%
pred, confidence = predict(image_path, modality)
print(f"Prediction: {'Anomaly' if pred == 1 else 'Normal'}")
print(f"Confidence: {confidence:.4f}")


# %%
# Quick smoke test: dummy forward pass to validate shapes
import torch
with torch.no_grad():
    dummy = torch.randn(1, 3, 128, 128)  # batch 1, RGB, 128x128
    out = ct_model(dummy)
    print('ct_model output shape:', out.shape)


