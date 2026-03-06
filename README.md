# ♻️ RecycleVision: AI-Powered Garbage Classification System

RecycleVision is a deep learning-based image classification system that automatically identifies different types of waste materials using Artificial Intelligence.

The project uses **Transfer Learning with EfficientNetB0** to classify garbage images into six recyclable categories.

This system can help improve **waste sorting, recycling efficiency, and environmental sustainability**.

---

# 📊 Project Overview

This project builds a computer vision model that can classify waste images into different categories such as cardboard, glass, plastic, etc.

The trained model is integrated into a **Streamlit web application** that allows users to upload an image and instantly get the predicted waste category along with confidence scores.

---

# 🧠 Technologies Used

- Python
- TensorFlow / Keras
- EfficientNetB0 (Transfer Learning)
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Streamlit

---

# 📂 Dataset

This project uses the **Garbage Classification Dataset (TrashNet)**.

Dataset contains **2532 images** across **6 classes**:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

⚠️ Dataset is **not included in this repository** due to size limitations.

Download the dataset from Kaggle:

👉 https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification

---

# 📁 Dataset Folder Structure

After downloading the dataset, place it inside the project directory like this:

```
RecycleVision_Project/

dataset/
└── Garbage classification/
├── cardboard
├── glass
├── metal
├── paper
├── plastic
└── trash
```


---

# 🚀 Features

- Garbage image classification using deep learning
- Transfer learning using EfficientNetB0
- Data augmentation
- Class imbalance handling using class weights
- Model evaluation with confusion matrix
- Streamlit web application for real-time prediction
- Confidence score visualization
- Sample images for testing

---

# 📈 Model Performance

**Model:** EfficientNetB0 (Transfer Learning)

**Validation Accuracy:** ~83–85%

Evaluation Metrics Used:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---

# 🖥️ Run The Project Locally

## 1️⃣ Clone Repository

👉 git clone: https://github.com/rudra-barman/RecycleVision-Project.git

cd RecycleVision_Project

---

## 2️⃣ Install Dependencies
```
pip install -r requirements.txt
```
---

## 3️⃣ Run Streamlit App
```
streamlit run app.py
```
---

# 🧪 Testing The Model

You can test the model using images in the **sample_images** folder.

Example images:
```
sample_images/

cardboard.jpg
glass.jpg
metal.jpg
paper.jpg
plastic.jpg
trash.jpg
```
---

# 📊 Project Workflow

1. Data Loading  
2. Exploratory Data Analysis (EDA)  
3. Data Preprocessing & Augmentation  
4. Transfer Learning Model (EfficientNetB0)  
5. Model Training  
6. Fine-Tuning  
7. Model Evaluation  
8. Streamlit Web Application  

---

# 📂 Project Structure
```
RecycleVision_Project/

app.py
requirements.txt
RecycleVision_Project.ipynb
RecycleVision_Final_Model.keras
RecycleVision_Presentation.pptx

sample_images/
├── cardboard.jpg
├── glass.jpg
├── metal.jpg
├── paper.jpg
├── plastic.jpg
└── trash.jpg
```
---

# 👨‍💻 Author

**Rudra Barman**

---

# 🌍 Future Improvements

- Deploy the application on Streamlit Cloud
- Improve model accuracy with larger dataset
- Add real-time camera waste detection
- Integrate with smart recycling systems

---




