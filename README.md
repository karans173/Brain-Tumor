# Brain Tumor MRI Classification with Explainable AI

This project is a complete deep learning pipeline and web application for classifying brain tumors from MRI scans. It leverages a transfer learning approach with the Xception architecture and provides model explanations through Grad-CAM visualizations.

<img width="1920" height="1080" alt="Screenshot (111)" src="https://github.com/user-attachments/assets/a1acfa5e-5c8f-4fcc-8c89-97c377767e15" />

## 📖 Project Overview

The goal of this project is to accurately classify brain MRI scans into one of four categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**. The project is structured into two main components:

1.  A **modular training pipeline** (`src/`) for data preprocessing, model building, training, and evaluation.
2.  A **Flask web application** (`app/`) that serves the trained model through a user-friendly interface, allowing users to upload an MRI and receive a prediction with explainability.

## ✨ Features

-   **High-Accuracy Classification:** Utilizes the powerful Xception model pre-trained on ImageNet for effective feature extraction.
-   **Interactive Web Interface:** A clean and simple UI built with Flask for easy image uploads and results visualization.
-   **Explainable AI (XAI):** Generates Grad-CAM (Gradient-weighted Class Activation Mapping) heatmaps to visualize which parts of the MRI the model focused on to make its prediction.
-   **Confidence Scores:** Displays a probability distribution chart showing the model's confidence for each tumor class.
-   **Prediction History:** Logs recent predictions to a CSV file and displays them in the application.
-   **Modular & Reusable Code:** The training pipeline is structured into separate modules for configuration, data processing, and model definition, following best practices in software engineering.

## 🛠️ Technology Stack

-   **Backend:** Python, Flask
-   **Deep Learning:** TensorFlow, Keras
-   **Data Manipulation:** Pandas, NumPy, Scikit-learn
-   **Image Processing:** OpenCV, Pillow
-   **Frontend:** HTML, CSS, Chart.js

## 📂 Project Structure

The repository is organized to separate the training pipeline from the web application, ensuring a clean and maintainable codebase.

```
brain_tumor_project/
├── app/                  # Flask application source code
│   ├── static/
│   ├── templates/
│   ├── app.py
│   └── requirements.txt
├── data/                 # (Not in repo) Dataset folder
├── models/               # (Not in repo) Saved Keras models
├── notebooks/            # Original Jupyter Notebook for exploration
├── src/                  # Modularized training pipeline
│   ├── config.py
│   ├── data_preprocessing.py
│   ├── model.py
│   └── train.py
├── .gitignore            # Files to be ignored by Git
└── README.md             # This file
```

## 🚀 Setup and Usage

Follow these steps to set up and run the project locally.

### Prerequisites

-   Python 3.9+
-   Git

### 1. Clone the Repository

```bash
git clone [https://github.com/YOUR_USERNAME/Brain-Tumor.git](https://github.com/YOUR_USERNAME/Brain-Tumor.git)
cd Brain-Tumor
```

### 2. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

```bash
# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 3. Install Dependencies

The required packages for the Flask application are listed in `app/requirements.txt`.

```bash
pip install -r app/requirements.txt
```

### 4. Download the Dataset

The model was trained on the "Brain MRI Images for Brain Tumor Detection" dataset from Kaggle.

-   **Download Link:** [https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
-   Create a `data/` directory in the project root.
-   Unzip the downloaded file and place the `Training` and `Testing` folders inside the `data/` directory.

### 5. Run the Training Pipeline (Optional)

If you wish to train the model from scratch, run the main training script. This will process the data, build and train the model, and save the final `Brain_Tumor_Classifier.keras` file into a `models/` directory.

```bash
python src/train.py
```

### 6. Run the Flask Application

Make sure a trained model (e.g., `Model-final.keras`) is present in the `models/` directory. Then, run the Flask app.

```bash
python app/app.py
```

Open your web browser and navigate to **`http://127.0.0.1:5000`** to use the application.

## 🧠 Model Architecture

The classification model is built using the Keras Functional API.

-   **Base Model:** Xception, pre-trained on ImageNet, with its top classification layers removed (`include_top=False`). The base model's layers are frozen to leverage their learned features.
-   **Pooling Layer:** A `GlobalMaxPooling2D` layer is used to reduce the spatial dimensions of the feature maps.
-   **Custom Head:** A custom classification head is added on top of the base model, consisting of:
    -   A `Dropout` layer (rate=0.3) for regularization.
    -   A `Dense` layer with 128 units and a ReLU activation function.
    -   A final `Dropout` layer (rate=0.25).
    -   An output `Dense` layer with 4 units (for the 4 classes) and a `softmax` activation function for probability distribution.
