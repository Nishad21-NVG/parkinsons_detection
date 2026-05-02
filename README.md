<<<<<<< HEAD
# Parkinsons Detection Project 
=======
# 🧠 Multi-Modal Parkinson’s Disease Detection System
### Final Year Capstone Project | Machine Learning + Deep Learning + Computer Vision

---

## 📌 Project Overview
Parkinson’s Disease is a neurodegenerative disorder that affects movement, speech, and motor skills. Early detection is very important for proper treatment and improving the quality of life of patients.

This project presents a **Multi-Modal Parkinson’s Disease Detection System** that uses **Voice Analysis, Spiral Drawing Image Analysis, and Video Motion Analysis** to detect Parkinson’s disease using Machine Learning, Deep Learning, and Computer Vision techniques.

The system integrates multiple AI models into a single web application using **Flask (Backend)** and **Streamlit (Frontend)**.

---

## 🎯 Objectives
- Detect Parkinson’s disease using AI/ML techniques.
- Use multiple data modalities for higher accuracy.
- Build a user-friendly web interface for prediction.
- Assist doctors in early diagnosis.
- Provide automated Parkinson’s screening system.

---

## 🧠 Models Used

| Modality | Algorithms Used |
|----------|----------------|
| 🎙️ Voice Analysis | Logistic Regression, SVM, Random Forest |
| 🖼️ Spiral Image Analysis | Convolutional Neural Network (CNN) |
| 🎥 Video Analysis | OpenCV Feature Extraction + Random Forest |

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Programming | Python |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Computer Vision | OpenCV |
| Backend | Flask |
| Frontend | Streamlit |
| Data Processing | NumPy, Pandas |
| Visualization | Matplotlib |

---

## 🏗️ System Architecture
                     +-------------------+
                     |       User        |
                     +---------+---------+
                               |
                +--------------+--------------+
                |              |              |
          +-----v-----+  +-----v-----+  +-----v-----+
          |   Voice   |  |   Image   |  |   Video   |
          |  Input    |  |  Input    |  |  Input    |
          +-----+-----+  +-----+-----+  +-----+-----+
                |              |              |
                v              v              v
      +----------------+ +----------------+ +----------------------+
      | Voice ML Model | | CNN Image Model| | Video Feature Extract|
      | LR, SVM, RF    | | TensorFlow     | | OpenCV               |
      +--------+-------+ +--------+-------+ +----------+-----------+
               |                  |                     |
               +------------------+---------------------+
                                  |
                                  v
                       +----------------------+
                       |   Prediction Engine  |
                       +----------+-----------+
                                  |
                                  v
                          +---------------+
                          | Flask Backend |
                          +-------+-------+
                                  |
                                  v
                       +--------------------+
                       | Streamlit Frontend |
                       +--------------------+
                                  |
                                  v
                          +---------------+
                          |  Final Result |
                          | Parkinson / No|
                          +---------------+

## 📁 Project Structure

```
parkinsons_detection/
│
├── backend/
│   ├── app.py                    # Flask backend API
│   └── routes.py                 # API routes for prediction
│
├── frontend/
│   └── streamlit_app.py          # Streamlit user interface
│
├── ml_models/
│   ├── voice_model/
│   │   ├── train_voice_model.py
│   │   └── voice_model.pkl
│   │
│   ├── image_model/
│   │   ├── train_image_model.py
│   │   └── cnn_model.h5
│   │
│   └── video_model/
│       ├── process_video.py
│       └── video_model.pkl
│
├── utils/
│   ├── feature_extraction.py     # Feature extraction functions
│   ├── preprocessing.py          # Data preprocessing
│   └── helper_functions.py
│
├── data/
│   ├── voice_dataset/
│   ├── spiral_images/
│   └── video_dataset/
│
├── notebooks/
│   └── model_training.ipynb
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── .gitignore                    # Ignore unnecessary files
└── main.py                       # Main integration script
```

## ▶️ How to Run the Project

### 1. Clone Repository

git clone https://github.com/Nishad21-NVG/parkinsons_detection.git

cd parkinsons_detection


### 2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate


### 3. Install Requirements

pip install -r requirements.txt


### 4. Train Models

python ml_models/voice_model/train_voice_model.py
python ml_models/image_model/train_image_model.py
python ml_models/video_model/process_video.py


### 5. Run Backend

python backend/app.py


### 6. Run Frontend

streamlit run frontend/streamlit_app.py


### 7. Open Browser

http://localhost:8501


---

## 📊 Output
- Parkinson’s Detection Result
- Confidence Score
- Prediction Accuracy

---

## 📈 Future Scope
- Real-time webcam detection
- Cloud deployment
- Mobile app
- Improve CNN accuracy

---

## 👨‍💻 Author
Nishad Ghatage  
Final Year Computer Engineering Student  
2026

Siddhant Goyal  
Final Year Computer Engineering Student  
2026

Aditya Shinde  
Final Year Computer Engineering Student  
2026
>>>>>>> 9535154887c37886b63551184e3e2630d6d3d7db
>>>>>>> Testing GitHub Achievement Badge 🚀

