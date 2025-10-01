<!-- <p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p> -->
<p align="center"><h1 align="center">SCRAP_IDENTIFICATION</h1></p>
<p align="center">
	<em>End-to-End ML Pipeline for Real-Time Material Classification</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/Stormbreaker-2/Scrap_Identification?style=default&logo=opensourceinitiative&logoColor=white&color=24ff00" alt="license">
	<img src="https://img.shields.io/github/last-commit/Stormbreaker-2/Scrap_Identification?style=default&logo=git&logoColor=white&color=24ff00" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/Stormbreaker-2/Scrap_Identification?style=default&color=24ff00" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/Stormbreaker-2/Scrap_Identification?style=default&color=24ff00" alt="repo-language-count">
</p>
<br>

## 🔗 Table of Contents

- [📍 Overview](#-overview)
- [👾 Features](#-features)
- [📁 Project Structure](#-project-structure)
  - [📂 Project Index](#-project-index)
- [🚀 Getting Started](#-getting-started)
  - [☑️ Prerequisites](#-prerequisites)
  - [⚙️ Installation](#-installation)
  - [🤖 Usage](#-usage)
  - [🧪 Testing](#-testing)
- [📊 Model Performance](#-model-performance)
- [🎯 Deployment Strategy](#-deployment-strategy)
- [📌 Project Roadmap](#-project-roadmap)
- [🔰 Contributing](#-contributing)
- [🎗 License](#-license)
- [🙌 Acknowledgments](#-acknowledgments)

---

## 📍 Overview

**Scrap_Identification** is an end-to-end machine learning pipeline designed to simulate real-time scrap material classification using computer vision. Built as a demonstration of production-ready ML systems, this project classifies waste materials into six categories: cardboard, glass, metal, paper, plastic, and trash. The system leverages transfer learning with ResNet18, implements a lightweight deployment strategy using TorchScript, and simulates a conveyor belt system for real-time inference with active learning capabilities.

**Key Highlights:**
- CNN-based classification using ResNet18 with transfer learning
- Real-time simulation loop mimicking industrial conveyor systems
- Active learning pipeline for continuous model improvement
- Lightweight deployment with model optimization
- Comprehensive evaluation metrics and visualization

---

## 👾 Features

### 🎯 Core Capabilities
- **Multi-Class Classification**: Identifies 6 material types with high accuracy
- **Transfer Learning**: Leverages pre-trained ResNet18 for efficient training
- **Data Augmentation**: Robust preprocessing pipeline with augmentation techniques
- **Real-Time Simulation**: Conveyor belt simulation for frame-by-frame classification
- **Confidence Thresholding**: Automatic flagging of low-confidence predictions
- **Active Learning**: Manual override and retraining queue for misclassifications

### 🛠 Technical Features
- **Model Optimization**: TorchScript conversion for deployment
- **Stratified Splitting**: Balanced train/validation/test splits
- **Class Balancing**: Weighted sampling to handle class imbalance
- **Fine-Tuning Strategy**: Layer-specific learning rates for optimal convergence
- **Comprehensive Metrics**: Accuracy, precision, recall, confusion matrices
- **Visualization Suite**: Training curves, confusion matrices, and performance plots

---

## 📁 Project Structure

```sh
└── Scrap_Identification/
    ├── LICENSE
    ├── README.md
    ├── data/
    │   ├── conveyor_samples/          # Sample images for simulation
    │   ├── test/                      # Test dataset (20% split)
    │   │   ├── cardboard/
    │   │   ├── glass/
    │   │   ├── metal/
    │   │   ├── paper/
    │   │   ├── plastic/
    │   │   └── trash/
    │   └── train/                     # Training dataset (80% split)
    │       ├── cardboard/
    │       ├── glass/
    │       ├── metal/
    │       ├── paper/
    │       ├── plastic/
    │       └── trash/
    ├── models/
    │   ├── inference/
    │   │   └── resnet18_scrap_ts.pt  # TorchScript model
    │   ├── resnet18_best_finetuned.pth  # Best model checkpoint
    │   └── resnet18_scrap.pth        # Base trained model
    ├── plots/
    │   ├── acc_curve.png             # Training/validation accuracy
    │   ├── loss_curve.png            # Training/validation loss
    │   ├── confusion_matrix.png      # Validation confusion matrix
    │   └── confusion_matrix_test.png # Test confusion matrix
    ├── results/
    │   ├── active_learning/          # Misclassified images for retraining
    │   ├── misclassified/            # Test set errors
    │   ├── review/                   # Low-confidence predictions
    │   ├── simulation_results.csv    # Basic simulation output
    │   └── simulation_results_bonus.csv  # Enhanced simulation with manual override
    ├── requirements.txt
    └── src/
        ├── train_improved.py         # Main training script with fine-tuning
        ├── split_dataset.py          # Dataset preparation utility
        ├── inference.py              # Single image inference
        ├── simulation.py             # Real-time conveyor simulation
        └── bonus_simulation.py       # Enhanced simulation with active learning
```

### 📂 Project Index

<details open>
	<summary><b><code>SCRAP_IDENTIFICATION/</code></b></summary>
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/requirements.txt'>requirements.txt</a></b></td>
				<td>Dependencies for PyTorch, torchvision, Pillow, scikit-learn, matplotlib, seaborn, and colorama</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details>
		<summary><b>src</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/src/train_improved.py'>train_improved.py</a></b></td>
				<td>Complete training pipeline with stratified splitting, weighted sampling, fine-tuning, and comprehensive evaluation</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/src/split_dataset.py'>split_dataset.py</a></b></td>
				<td>Utility script to split TrashNet dataset into 80/20 train/test splits with class preservation</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/src/inference.py'>inference.py</a></b></td>
				<td>Standalone inference script for single image classification with confidence scores</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/src/simulation.py'>simulation.py</a></b></td>
				<td>Real-time conveyor simulation with confidence thresholding and CSV logging</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/src/bonus_simulation.py'>bonus_simulation.py</a></b></td>
				<td>Enhanced simulation with manual override, review queue, and active learning dataset generation</td>
			</tr>
			</table>
		</blockquote>
	</details>
	<details>
		<summary><b>models</b></summary>
		<blockquote>
			<table>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/models/resnet18_scrap.pth'>resnet18_scrap.pth</a></b></td>
				<td>Base trained model checkpoint</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/models/resnet18_best_finetuned.pth'>resnet18_best_finetuned.pth</a></b></td>
				<td>Best performing model checkpoint (selected based on validation accuracy)</td>
			</tr>
			<tr>
				<td><b><a href='https://github.com/Stormbreaker-2/Scrap_Identification/blob/master/models/inference/resnet18_scrap_ts.pt'>resnet18_scrap_ts.pt</a></b></td>
				<td>Optimized TorchScript model for lightweight deployment</td>
			</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## 🚀 Getting Started

### ☑️ Prerequisites

Before getting started with Scrap_Identification, ensure your runtime environment meets the following requirements:

- **Programming Language:** Python 3.8+
- **Package Manager:** Pip
- **Hardware:** CUDA-capable GPU (optional, but recommended for training)
- **Operating System:** Windows/Linux/MacOS

### ⚙️ Installation

Install Scrap_Identification using the following steps:

**1. Clone the repository:**
```sh
git clone https://github.com/Stormbreaker-2/Scrap_Identification
```

**2. Navigate to the project directory:**
```sh
cd Scrap_Identification
```

**3. Install dependencies:**
```sh
pip install -r requirements.txt
```

**4. Download the dataset:**

This project uses the [TrashNet dataset](https://github.com/garythung/trashnet). Download and extract it, then run:

```sh
python src/split_dataset.py
```

This will automatically organize the data into train/test splits in the `data/` directory.

---

### 🤖 Usage

#### **Training the Model**

To train the ResNet18 model with fine-tuning and data augmentation:

```sh
python src/train_improved.py
```

**Training Features:**
- Stratified 85/15 train/validation split
- Weighted sampling for class balance
- Layer-specific learning rates (Layer 4: 1e-4, FC: 1e-3)
- Class-weighted loss function
- StepLR scheduler (step=7, gamma=0.1)
- Early stopping based on validation accuracy
- Automatic saving of best model checkpoint

**Expected Output:**
- Best model saved to `models/resnet18_best_finetuned.pth`
- Training curves in `plots/` (loss, accuracy)
- Test confusion matrix in `plots/confusion_matrix_test.png`
- Misclassified images in `results/misclassified/`

#### **Single Image Inference**

To classify a single image:

```sh
python src/inference.py
```

Edit the `test_img` path in the script to point to your image.

**Example Output:**
```
Using device: cuda
✅ Best fine-tuned model loaded successfully!
Prediction: plastic (94.23% confidence)
```

#### **Real-Time Simulation**

**Basic Simulation:**
```sh
python src/simulation.py
```

Processes all images in `data/test/`, logs predictions to CSV, and flags low-confidence predictions (<70%).

**Enhanced Simulation (with Active Learning):**
```sh
python src/bonus_simulation.py
```

**Additional Features:**
- Interactive manual override for misclassifications
- Automatic copying of low-confidence images to review folder
- Active learning queue generation for retraining
- Color-coded console output (green: confident, yellow: low confidence)

**Example Output:**
```
[plastic47.jpg] → plastic (94.23%)
[glass12.jpg] → glass (68.45%) LOW CONFIDENCE
Moved glass12.jpg to review folder.
Is prediction correct? (y/n) for glass12.jpg: n
Enter correct class: cardboard
Added glass12.jpg to active learning queue under cardboard
```

---

### 🧪 Testing

The model is evaluated on a held-out test set (20% of data) with the following metrics:

**Run evaluation:**
```sh
python src/train_improved.py
```

**Evaluation includes:**
- Classification report (precision, recall, F1-score per class)
- Confusion matrix visualization
- Misclassified images saved with true/predicted labels
- Per-class performance breakdown

---

## 📊 Model Performance

### Architecture Details

**Base Model:** ResNet18 (pre-trained on ImageNet)

**Fine-Tuning Strategy:**
- Freeze all layers except Layer 4 and FC layer
- Layer 4 learning rate: 1e-4
- FC layer learning rate: 1e-3
- Optimizer: SGD with momentum (0.9) and weight decay (1e-4)
- Scheduler: StepLR (step=7, gamma=0.1)

**Training Configuration:**
- Epochs: 20
- Batch size: 32
- Input size: 224×224
- Weighted random sampling for class balance
- Class-weighted cross-entropy loss

### Data Augmentation

**Training Augmentation:**
- Random resized crop (scale: 0.8-1.0)
- Random horizontal flip
- Color jitter (brightness, contrast, saturation: ±20%)
- Random rotation (±15°)
- ImageNet normalization

**Validation/Test Transform:**
- Resize to 256
- Center crop to 224
- ImageNet normalization

### Dataset Information

**Source:** TrashNet Dataset

**Why TrashNet?**
- Real-world waste classification problem
- Diverse material types commonly found in recycling
- Sufficient samples per class for robust training
- Publicly available and well-documented
- Directly applicable to industrial sorting applications

**Classes:** 6 material types
- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash (general waste)

**Split:**
- Training: 80% (with 15% internal validation)
- Testing: 20%

**Total Images:** ~2,500 images across all classes

### Performance Metrics

Training produces comprehensive metrics including:
- Per-epoch training/validation loss and accuracy
- Final test set classification report
- Confusion matrix visualization
- Misclassification analysis with image storage

Results are saved in:
- `plots/acc_curve.png` - Training accuracy progression
- `plots/loss_curve.png` - Loss curves
- `plots/confusion_matrix_test.png` - Test set confusion matrix
- `results/misclassified/` - All incorrectly classified test images

---

## 🎯 Deployment Strategy

### Model Optimization

**TorchScript Conversion:**
The trained model can be converted to TorchScript for production deployment:

```python
import torch
model_scripted = torch.jit.script(model)
model_scripted.save('models/inference/resnet18_scrap_ts.pt')
```

**Benefits:**
- Platform-independent inference
- Optimized for production environments
- Compatible with C++ deployment
- Reduced overhead compared to Python inference

### Deployment Considerations

**Current Implementation:**
- PyTorch-based inference
- CUDA acceleration (when available)
- Batch processing capability
- Real-time simulation loop

**Production Recommendations:**
- Use TorchScript model for faster inference
- Implement model quantization for edge devices
- Deploy on edge hardware (Jetson Nano/Xavier) for real-time sorting
- Set up monitoring for prediction confidence distribution
- Implement periodic model retraining with active learning queue

### Edge Device Compatibility

The model architecture and size make it suitable for deployment on:
- NVIDIA Jetson Nano
- NVIDIA Jetson Xavier
- Raspberry Pi 4 (with optimization)
- Industrial PCs with GPU acceleration

**Optimization Tips for Edge:**
- Use TorchScript or ONNX format
- Apply INT8 quantization
- Reduce input resolution if needed (trade-off with accuracy)
- Batch multiple frames for throughput optimization

---

## 📌 Project Roadmap

- [X] **Dataset Preparation** - Split TrashNet into train/test sets
- [X] **Model Development** - ResNet18 with transfer learning
- [X] **Training Pipeline** - Fine-tuning with data augmentation
- [X] **Evaluation Metrics** - Classification report and confusion matrix
- [X] **Inference Script** - Single image classification
- [X] **Real-Time Simulation** - Conveyor belt simulation loop
- [X] **Active Learning** - Manual override and retraining queue
- [ ] **ONNX Conversion** - Cross-platform deployment format
- [ ] **Jetson Deployment** - Edge device testing and optimization
- [ ] **Model Quantization** - INT8 optimization for faster inference
- [ ] **Web Interface** - Simple UI for real-time classification
- [ ] **Performance Monitoring** - Logging and drift detection

---

## 🔰 Contributing

Contributions are welcome! Here's how you can help:

- **💬 [Join the Discussions](https://github.com/Stormbreaker-2/Scrap_Identification/discussions)**: Share insights, provide feedback, or ask questions
- **🐛 [Report Issues](https://github.com/Stormbreaker-2/Scrap_Identification/issues)**: Submit bugs or log feature requests
- **💡 [Submit Pull Requests](https://github.com/Stormbreaker-2/Scrap_Identification/pulls)**: Review open PRs and submit your own

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project to your GitHub account
2. **Clone Locally**: Clone the forked repository to your local machine
   ```sh
   git clone https://github.com/Stormbreaker-2/Scrap_Identification
   ```
3. **Create a New Branch**: Always work on a new branch with a descriptive name
   ```sh
   git checkout -b feature/your-feature-name
   ```
4. **Make Your Changes**: Develop and test your changes locally
5. **Commit Your Changes**: Commit with a clear, descriptive message
   ```sh
   git commit -m 'Add: Brief description of your changes'
   ```
6. **Push to GitHub**: Push the changes to your forked repository
   ```sh
   git push origin feature/your-feature-name
   ```
7. **Submit a Pull Request**: Create a PR against the original repository with a detailed description
8. **Review**: Once approved, your PR will be merged. Thank you for contributing!

</details>

---

## 🎗 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgments

**Dataset:**
- [TrashNet](https://github.com/garythung/trashnet) by Gary Thung and Mindy Yang - Public dataset for waste classification research

**Frameworks & Libraries:**
- PyTorch and torchvision for deep learning infrastructure
- scikit-learn for evaluation metrics and data splitting
- Matplotlib and Seaborn for visualization
- Colorama for enhanced console output

**Inspiration:**
- This project was developed as a demonstration of end-to-end ML pipeline development, from data preparation to deployment-ready inference, applicable to real-world industrial sorting systems.

**Assignment Context:**
- Developed as part of the AlfaStack Machine Learning Internship assignment to showcase practical ML engineering skills including model development, optimization, deployment awareness, and documentation.

---

<p align="center">
  <em>Built with ❤️ for sustainable waste management through AI</em>
</p>
