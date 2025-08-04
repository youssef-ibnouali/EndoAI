# Welcome to EndoAI 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

> **EndoAI** is an AI-based gastric endoscopy analysis platform that uses deep learning for image classification, integrated into a full-stack web application built with Flask and React.

---

## 🚀 Features

- 🔬 Deep Learning Classification: CNN and Transformer models trained on real annotated gastric endoscopy data.
- 🧠 Patch-wise Analysis: Sliding-window patch selection with entropy, sharpness, and tissue masking.
- 🧾 PDF Report Generation: Automatically summarizes class proportions and diagnosis results.
- 🖼️ Visual Overlay: Class predictions visualized as bounding boxes with color-coded labels on original images.
- 🌐 Full-Stack Application: Frontend in React (Vite), backend API in Flask, with clean user interface and upload flow.
- 🧪 Model Benchmarking: Easily compare over 20+ architectures (EfficientNet, ResNet, ViT, Swin, etc.) with saved metrics and plots.
- 👨‍⚕️ Medical Context Integration: MATLAB scripts support pre-selection of informative regions using entropy and morphological filtering.
- 👥 Login System: Role-based access using organization-linked credentials (defined in users.json).
- 🧰 Custom Training Pipeline: Includes support for patch augmentation, early stopping, class balancing, and confusion matrix generation.

---

## 🛠️ Install

### Prerequisites 
Make sure you have:

- Python 3.10+
- Node.js & npm

### clone the repository

```bash
git clone https://github.com/youssef-ibnouali/EndoAI.git
cd EndoAI
```

### Install dependencies

Install backend and frontend dependencies:

```bash
# Backend
pip install -r requirements.txt

# Frontend
cd frontend
npm install
```

## 💻 Usage

```sh
cd frontend
npm run start-all
```

This will start:

- 🔁 Flask API at: `http://localhost:5000`
- ⚛️ React frontend at: `http://localhost:5173`

> **Note**: The backend loads model weights and serves classification & report-generation endpoints.

---

## 🧠 Model Info

- The model is a CNN trained using PyTorch.
- Patch extraction, augmentation, and inference logic lives in:  
  `train_cnn/classify_nbi_image.py`

You can retrain models using the scripts in `train_cnn/` or load new weights.

---

## 🧾 Project Structure

```
EndoAI/
├── app.py                        # Flask backend (main API)
├── users.json                    # Static login data (org/user/pass)
├── requirements.txt              # Python dependencies
├── README.md
├── frontend/                     # React frontend
│   ├── src/
│   │   ├── assets/               # Images, logos
│   │   └── components/           # Pages and components (Home, Upload, Report, etc.)
│   ├── public/
│   ├── package.json
│   └── vite.config.js
├── results/                      # Stores processed results (annotated images, predictions, plots, metrics, etc.)
│   ├── trained_models/           # Saved PyTorch model checkpoints (.pth)
│   ├── plots/                    # Training and evaluation plots
│   ├── model_compar/             # Plots comparing different models
│   └── ...
├── uploads/                      # Temporarily stores uploaded endoscopic images
├── train_cnn/                    # Core model training and inference logic
│   ├── model.py                  # CNN and transformer model definitions
│   ├── train.py                  # Training loop with early stopping, metrics
│   ├── classify_nbi_image.py     # Patch extraction and classification pipeline
│   ├── predict_batch.py          # Python-based batch prediction (used via MATLAB)
│   ├── try_models.py             # Batch test multiple models on your dataset
│   ├── data/
│   │   ├── raw/                  # Raw patches before augmentation
│   │   ├── processed/            # Train/val/test split directories
│   │   └── augmentation.py       # Data augmentation logic
│   └── utils/                    # Optional: metrics, plots, etc.
├── matlab/                       # MATLAB scripts for integration and visualization
│   ├── classify_nbi_image_cnn.m  # Calls Python classifier, draws detections
│   ├── entropy_selection.m       # Patch selection using entropy and masks
│   └── ...
```

---

## 📁 Dataset Format

Training/validation/test datasets should be structured like:

```
train_cnn/data/processed/
├── train/
│   ├── AG/
│   ├── Cancer/
│   ├── Dysplasia/
│   ├── IM/
│   └── Normal/
├── val/
├── test/
```
Each folder contains patch images for the corresponding class.

---

## 💡 How to Extend

To add more models, simply define them in `train_cnn/model.py` using the `get_model()` function pattern. You can test new models by adding their names to `models = [...]` in `try_models.py`.

---

## 🌍 API Endpoints (Flask)

| Endpoint             | Method | Description                          |
|----------------------|--------|--------------------------------------|
| `/upload`            | POST   | Uploads and classifies an image      |
| `/classify`          | POST   | Classifies uploaded image patches    |
| `/report`            | GET    | Generates the PDF report             |
| `/organizations`     | GET    | Lists available organizations        |
| `/login`             | POST   | Validates login via `users.json`     |

---

## 🧪 Example Users

Stored in `users.json`:

```json
{
  "users": [
    {
      "organization": "Military Technical Academy 'Ferdinand I' of Bucharest",
      "username": "admin",
      "password": "admin"
    }
  ]
}
```

---

## 🧪 Test the Pipeline

To test the full classification and report generation pipeline:

1. Start backend and frontend
2. Upload a `.jpg` or `.png` gastric endoscopic image
3. Wait for patch-wise classification and PDF report

Output will include:
- Overlaid image with color-coded class boxes
- Classification percentages
- Downloadable PDF report

---

## Author

👤 **Youssef IBNOUALI**

* Github: [@youssef-ibnouali](https://github.com/youssef-ibnouali)
* LinkedIn: [@youssef-ibnouali](https://linkedin.com/in/youssef-ibnouali)

---

## 📄 License :

This project is licensed under the MIT License.

---

## 📣 Citation / Credit

If you use this platform, are inspired by it, or extend it in your work or academic project, please give proper attribution:

> "This project was built upon [EndoAI](https://github.com/youssef-ibnouali/EndoAI) by Youssef IBNOUALI."

You may also cite it like this:

```
@misc{EndoAI2025,
  author = {Youssef IBNOUALI},
  title = {EndoAI: Deep Learning Platform for Gastric Endoscopy},
  year = {2025},
  url = {https://github.com/youssef-ibnouali/EndoAI}
}
```