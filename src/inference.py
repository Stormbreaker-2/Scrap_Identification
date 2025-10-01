import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

model = resnet18(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, len(classes))

model.load_state_dict(torch.load(
    r"d:\AlfaStack Assignment\models\resnet18_best_finetuned.pth",
    map_location=device
))
model = model.to(device)
model.eval()
print("✅ Best fine-tuned model loaded successfully!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"❌ Error opening image {img_path}: {e}")
        return None, None

    img_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_class = torch.max(probs, 0)

    return classes[pred_class.item()], confidence.item()
if __name__ == "__main__":
    test_img = r"d:\AlfaStack Assignment\data\test\plastic\plastic47.jpg"

    label, conf = predict_image(test_img)
    if label:
        print(f"Prediction: {label} ({conf*100:.2f}% confidence)")
