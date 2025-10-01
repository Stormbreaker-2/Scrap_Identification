import os
import shutil
import random

dataset_dir = r"D:/AlfaStack Assignment/trashnet-master/data/dataset-resized"
output_dir = r"D:/AlfaStack Assignment/data"

classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
for split in ["train", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_dir, split, cls), exist_ok=True)

for cls in classes:
    img_dir = os.path.join(dataset_dir, cls)
    images = os.listdir(img_dir)
    random.shuffle(images)

    split_point = int(0.8 * len(images))
    train_imgs = images[:split_point]
    test_imgs = images[split_point:]

    for img in train_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(output_dir, "train", cls, img))

    for img in test_imgs:
        shutil.copy(os.path.join(img_dir, img), os.path.join(output_dir, "test", cls, img))

print("✅ Dataset split into train and test folders successfully!")
