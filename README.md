# Helmet Detection

This project is a web application for detecting whether a person is wearing a helmet in an image or a live video stream. It uses a YOLOv8 model for object detection.

## Features

- **Image-based Detection:** Upload an image to detect helmets.
- **Real-time Detection:** Use your webcam for live helmet detection.
- **Detailed Results:** Get detailed information about the detections, including bounding boxes, confidence scores, and class labels.

## Tech Stack

- **Backend:** Python, FastAPI, YOLOv8, PyTorch
- **Frontend:** React, TypeScript

## Project Structure

```
.
├── backend
│   ├── app
│   │   ├── __init__.py
│   │   └── main.py
│   ├── requirements.txt
│   └── yolov8m.pt
├── frontend
│   ├── public
│   ├── src
│   ├── package.json
│   └── tsconfig.json
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- Node.js and npm
- Git

### Backend Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd helmet-detection/backend
   ```

2. **Create a virtual environment and activate it:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Model Path Configuration:**

   **Important:** The model path in `backend/app/main.py` is currently hardcoded to an absolute path. You need to change it to a relative path to make the project portable.

   Open `backend/app/main.py` and change the `model_path` variable to:
   ```python
   # OLD
   # model_path = '/Users/mateonegri/Developer/sist-inteligentes/runs/runs/detect/helmet_detection_v1_single_gpu3/weights/best.pt'

   # NEW
   model_path = 'yolov8m.pt' 
   ```
   Or to the correct path of your model, assuming it is inside the `backend` folder. The repository contains a `yolov8m.pt` file in the `backend` directory. If this is the model you want to use, the path should be `'yolov8m.pt'`.

### Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd ../frontend
   ```

2. **Install the dependencies:**
   ```bash
   npm install
   ```

## Usage

1. **Run the backend server:**
   - Make sure you are in the `backend` directory with the virtual environment activated.
   - Run the following command:
     ```bash
     uvicorn app.main:app --host 0.0.0.0 --port 8080
     ```
   - The API will be available at `http://localhost:8080`.

   **Note:** The frontend expects the backend to be running on port `8080`. If you run the backend on a different port, you will need to update the fetch URL in `frontend/src/App.tsx`.

2. **Run the frontend application:**
   - In a new terminal, navigate to the `frontend` directory.
   - Run the following command:
     ```bash
     npm start
     ```
   - The application will open in your browser at `http://localhost:3000`.

Now you can upload an image or use the live detection feature to see the helmet detection in action. 