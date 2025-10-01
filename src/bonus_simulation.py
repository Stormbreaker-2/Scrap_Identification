import os
import csv
import shutil
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
import time
from colorama import Fore, Style, init

init(autoreset=True)

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
print("Model loaded for simulation with bonus features!")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def predict_image(img_path):
    image = Image.open(img_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, pred_class = torch.max(probs, 0)
    return classes[pred_class.item()], confidence.item()

if __name__ == "__main__":
    input_folder = r"d:\AlfaStack Assignment\data\test"
    output_csv = r"d:\AlfaStack Assignment\results\simulation_results_bonus.csv"
    review_folder = r"d:\AlfaStack Assignment\results\review"
    active_learning_folder = r"d:\AlfaStack Assignment\results\active_learning"

    os.makedirs(review_folder, exist_ok=True)
    os.makedirs(active_learning_folder, exist_ok=True)

    with open(output_csv, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Image", "PredictedClass", "Confidence", "FinalClass", "LowConfidenceFlag"])

        for root, dirs, files in os.walk(input_folder):
            for filename in files:
                if filename.lower().endswith((".jpg", ".png", ".jpeg")):
                    img_path = os.path.join(root, filename)
                    label, conf = predict_image(img_path)
                    low_flag = "LOW" if conf < 0.7 else ""
                    final_class = label

                    if low_flag:
                        print(Fore.YELLOW + f"[{filename}] → {label} ({conf*100:.2f}%) LOW CONFIDENCE")
                    else:
                        print(Fore.GREEN + f"[{filename}] → {label} ({conf*100:.2f}%)")

                    if low_flag:
                        review_path = os.path.join(review_folder, filename)
                        shutil.copy(img_path, review_path)
                        print(Fore.YELLOW + f"Moved {filename} to review folder.")

                        user_input = input(f"Is prediction correct? (y/n) for {filename}: ").strip().lower()
                        if user_input == "n":
                            print(f"Available classes: {classes}")
                            correct_class = input("Enter correct class: ").strip().lower()
                            if correct_class in classes:
                                final_class = correct_class
                                corrected_dir = os.path.join(active_learning_folder, correct_class)
                                os.makedirs(corrected_dir, exist_ok=True)
                                shutil.copy(img_path, corrected_dir)
                                print(Fore.CYAN + f"Added {filename} to active learning queue under {correct_class}")

                    writer.writerow([filename, label, f"{conf*100:.2f}", final_class, low_flag])
                    time.sleep(0.5)

    print(f"\nSimulation finished. Results saved at {output_csv}")
    print(f"Review images at {review_folder}")
    print(f"Active learning dataset at {active_learning_folder}")
