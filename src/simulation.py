import os
import csv
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import time

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
print("✅ Model loaded for simulation!")

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
    input_folder = r"d:\AlfaStack Assignment\data\test"
    output_csv = r"d:\AlfaStack Assignment\results\simulation_results.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:  # ✅ Added encoding="utf-8"
        writer = csv.writer(file)
        writer.writerow(["Image", "PredictedClass", "Confidence", "LowConfidenceFlag"])

        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(root, filename)

                    label, conf = predict_image(img_path)
                    if label:
                        low_flag = "⚠️" if conf < 0.7 else ""
                        print(f"[{filename}] → {label} ({conf*100:.2f}%) {low_flag}")

                        writer.writerow([filename, label, f"{conf*100:.2f}", low_flag])

                        time.sleep(0.5)

    print(f"\n✅ Simulation finished! Results saved at {output_csv}")
