🧠 Multi-Modal Parkinson’s Disease Detection System
Final Year Capstone Project | Machine Learning + Deep Learning + Computer Vision
📌 Project Overview

This project presents a Multi-Modal Parkinson’s Disease Detection System that analyzes voice recordings, spiral drawing images, and video data to detect Parkinson’s disease at an early stage using Machine Learning, Deep Learning, and Computer Vision techniques.

Traditional diagnosis of Parkinson’s disease is time-consuming and requires expert neurological examination. This system provides an AI-based automated screening tool that helps in early detection and risk assessment.

The system uses:

Voice Analysis
Spiral Drawing Image Analysis
Video Motion Analysis

All models are integrated into a Streamlit web application with a Flask backend.

🎯 Objectives
Detect Parkinson’s disease using AI/ML techniques.
Use multiple data modalities for higher accuracy.
Build a user-friendly web interface for prediction.
Assist doctors in early diagnosis.
🧠 Models Used
Modality	Algorithms Used
🎙️ Voice Analysis	Logistic Regression, SVM, Random Forest
🖼️ Spiral Image	Convolutional Neural Network (CNN)
🎥 Video Analysis	Random Forest + OpenCV Feature Extraction
🛠️ Tech Stack
Category	Technology
Programming	Python
Machine Learning	Scikit-learn
Deep Learning	TensorFlow / Keras
Computer Vision	OpenCV
Backend	Flask
Frontend	Streamlit
Data Processing	NumPy, Pandas
Visualization	Matplotlib
🏗️ System Architecture
User Input
   │
   ├── Voice Recording ──> ML Model ──┐
   ├── Spiral Image ─────> CNN Model ─┤──> Prediction Result
   └── Video Recording ──> CV + ML ───┘
                                │
                          Flask Backend
                                │
                          Streamlit Frontend
📂 Project Structure
parkinsons_detection/
│
├── backend/
│   └── app.py
│
├── frontend/
│   └── streamlit_app.py
│
├── ml_model/
│   ├── voice_model/
│   ├── image_model/
│   └── video_model/
│
├── utils/
├── requirements.txt
├── README.md
└── .gitignore
▶️ How to Run the Project
Step 1 — Install Requirements
pip install -r requirements.txt
Step 2 — Train Models
python train_voice_model.py
python train_image_model.py
python process_video.py
Step 3 — Start Backend
python backend/app.py
Step 4 — Start Frontend
streamlit run frontend/streamlit_app.py
📊 Expected Output

The system will display:

Prediction: Parkinson’s Detected / Not Detected
Confidence Score
Model Accuracy
📈 Future Improvements
Add real-time webcam detection
Deploy on cloud (AWS / Azure)
Improve CNN accuracy
Add medical report generation
Mobile application integration
👨‍💻 Author

Nishad Ghatage
Final Year Computer Engineering Student
Capstone Project – 2025-26

Siddhant Goyal
Final Year Computer Engineering Student
Capstone Project – 2025-26

Aditya Shinde
Final Year Computer Engineering Student
Capstone Project – 2025-26
