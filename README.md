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
├── train_cnn/                    # Core module for training, evaluation, and inference
│   ├── model.py                  # Definitions of all supported CNN and Transformer architectures
│   ├── train.py                  # Training loop with early stopping, weighted loss, and validation
│   ├── classify_nbi_image.py     # Semi-automatic pipeline: patch selection, filtering, scoring, and classification
│   ├── try_models.py             # Script to train and evaluate multiple models with metrics export
│   ├── predict_batch.py          # Optional: batch classification entry point (used from MATLAB or scripts)
│   ├── data_process.py           # Handles augmentation, elastic deformation, and dataset splitting
│   ├── extract_patchs.py         # GUI-assisted patch extraction and annotation from selected endoscopic image
│   ├── model_comparaison.py      # Compare trained models via global and per-class metrics and plots
│   ├── data/
│   │   ├── raw/                  # Human-labeled image patches (e.g. AG, IM, Normal, etc.)
│   │   ├── processed/            # Augmented and split dataset (train/val/test folders)
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

## 🧭 How to Train & Deploy Like Me

To follow the same development and experimentation process used in this project, here’s a 4-step pipeline:

### 1️⃣ Prepare Your Data

For detailed instructions on preparing your data, please refer to the documentation here:  
[Data Preparation Guide](https://uncloud.univ-nantes.fr/index.php/s/jkoNWpPajnCcpG3)


---

### 2️⃣ Select and Compare Models (optional)

If you're **testing several architectures**:

- **Add your model** to `train_cnn/model.py` using the same class structure as existing ones.
- **Add your model's name** to the `models = [...]` list inside `try_models.py`.
- **Run `try_models.py`** to automatically train and evaluate each model.
- **Run `model_comparaison.py`** to generate visual comparisons (bar plots + CSV summaries) and pick the best one based on `macro_f1`.

```bash
python train_cnn/try_models.py
python train_cnn/model_comparaison.py
```

> 🧠 If your model is pretrained, make sure to use the correct input size and load pretrained weights in your custom class.

---

### 3️⃣ Train Your Final Model

If you already selected a single model to train:

- **Go to `main.py`**
- Replace `model_name = ...` with your model's name
- Adjust training settings (epochs, batch size, learning rate...)
- Run the script to train and save your `.pth` model

```bash
python main.py
```

> This generates training plots, evaluation metrics, and a trained model file in `results/`.

---

### 4️⃣ Use the Model in Classification

To deploy your trained model in the **web platform**:

- Open `train_cnn/classify_nbi_image.py`
- Replace the value of `model_path` with the path to your new `.pth` file (e.g., `results/model_20250801_1732.pth`)
- The classification API will now use your updated model

```python
model_path = "results/model_date_hour.pth"
```

That’s it! You can now upload images via the frontend and get real-time patch-wise classification and PDF reports using your custom-trained model.

---

## 🌍 API Endpoints (Flask)

| Endpoint             | Method | Description                                                                  |
|----------------------|--------|------------------------------------------------------------------------------|
| `/classify`          | POST   | Uploads an image and returns patch-wise classification and scores           |
| `/result.png`        | GET    | Returns the annotated image with bounding boxes                             |
| `/generate_report`   | GET    | Generates and downloads a PDF report with diagnosis and patient history      |
| `/organizations`     | GET    | Returns available organizations from `users.json`                            |
| `/login`             | POST   | Authenticates user credentials using data from `users.json`                  |

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