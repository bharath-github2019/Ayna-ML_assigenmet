# Ayna-ML_assigenmet


# UNet Polygon Colorization – Ayna ML Internship Assignment

This project implements a UNet-based image segmentation model to colorize polygons in input images based on the specified color name. The model is trained using PyTorch and tracked using [Weights & Biases (wandb)](https://wandb.ai/site/).

---

## 📁 Folder Structure

```
.
├── UNet.py                   # UNet model architecture
├── train.py                  # Training script
├── dataset.py                # Custom dataset class (PolygonDataset)
├── trainings/
│   ├── inputs/               # Input images with polygons (e.g. triangle.png)
│   └── outputs/              # Ground truth colorized images (e.g. magenta_triangle.png)
├── validations/
│   ├── inputs/               # Validation inputs
│   └── outputs/              # Validation targets
├── train_model.ipynb         # Jupyter Notebook (optional training visualization)
├── requirements.txt
└── README.md                 # Project overview
```

---

🚀 Setup Instructions

1. Clone the repository** and navigate to the project folder:
   ```bash
   git clone https://github.com/bharath-github2019/Ayna-ML_assigenmet/
   cd UNet.py
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Weights & Biases (wandb):
   ```bash
   pip install wandb
   wandb login
   ```

4. Run Training:
   ```python
   from UNet import UNet
   from train import train_model
   from dataset import PolygonDataset
   from torch.utils.data import DataLoader
   import torch

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

   train_dataset = PolygonDataset("trainings/inputs", "trainings/outputs")
   val_dataset = PolygonDataset("validations/inputs", "validations/outputs")

   train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
   val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

   model = UNet(num_classes=10).to(device)
   train_model(model, train_loader, val_loader, num_epochs=10, device=device)
   ```

---

🧠 Model

The model used is a standard **UNet** architecture designed for semantic segmentation tasks. The final layer outputs class logits for each pixel, enabling colorization based on polygon types and color names.

---

📊 Tracking with wandb

Training metrics including loss curves are logged to Weights & Biases. Visit your wandb dashboard to visualize experiment runs:

👉 https://wandb.ai/

---

📦 Dependencies

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- wandb
- PIL

Install all via:
```bash
pip install -r requirements.txt
```

---

✅ Deliverables

- [x] UNet model implementation (`UNet.py`)
- [x] Training script with logging (`train.py`)
- [x] Custom dataset loader (`dataset.py`)
- [x] Training on provided inputs and outputs
- [x] wandb integration for training visualization
- [x] Clean, documented codebase and README

---

## ✨ Author

Bharath K  
ML Intern – Ayna  
2025
