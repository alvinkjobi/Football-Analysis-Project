# ⚽ Football Analysis Project

Welcome to the **Football Analysis Project** repository! This is a powerful MERN + Python-based web application designed for analyzing football images using advanced object detection with YOLOv8. The system identifies key features in uploaded images, processes them using a Python backend, and stores the results in MongoDB.

---

## 📚 Table of Contents

- [Features](#features)  
- [Tech Stack](#tech-stack)  
- [Installation](#installation)  
---

## ✨ Features

- 🧠 **YOLOv8 Integration**: Real-time football object detection using Ultralytics YOLOv8 model.
- 📸 **Image Upload & Processing**: Upload football-related images for automated analysis.
- 🗃️ **Base64 Storage**: Processed images and metadata are stored in MongoDB as base64 strings.
- 📈 **MERN Stack Dashboard**: Interactive frontend for viewing uploaded images and detection results.
- ⚙️ **Scalable Python Backend**: FastAPI or Flask for model execution and processing.

---

## 🛠️ Tech Stack

**Frontend:**
- React.js
- Axios
- React Router

**Backend (Node.js):**
- Express.js
- MongoDB + Mongoose
- Base64 image handling

**Backend (Python):**
- YOLOv8 (Ultralytics)
- OpenCV

---

## 🚀 Installation

### 1. Clone the repository
```bash
git clone https://github.com/alvinkjobi/Football-Analysis-Project.git
cd Football-Analysis-Project
npm install
npm run dev
