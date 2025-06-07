from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
from ultralytics import YOLO
import cv2
from app.class_fix_script import create_corrected_model

app = FastAPI(title="Helmet Detection API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Configuration ---
# Correct class names mapping
# The model was trained with swapped labels. This dictionary corrects them.
# 0 -> With_Helmet (Correct)
# 1 -> Without_Helmet (Correct)
CORRECTED_NAMES = {0: 'With_Helmet', 1: 'Without_Helmet'}

# Global variable to store the loaded model
model = None

def load_model():
    """Load the trained YOLOv8 helmet detection model"""
    global model
    try:
        # Load YOLOv8 model. The class names are corrected at runtime.
        model_path = '/Users/mateonegri/Developer/sist-inteligentes/runs/runs/detect/helmet_detection_v1_single_gpu3/weights/best.pt'
        model = create_corrected_model(model_path)  # This function loads the model
        print("YOLOv8 model loaded successfully! Class names will be corrected during processing.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

def process_yolo_results(results, confidence_threshold=0.5):
    """Process YOLOv8 detection results to determine helmet status"""
    with_helmet_detections = []
    without_helmet_detections = []
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Use the corrected class name from our mapping
                class_name = CORRECTED_NAMES.get(class_id, "Unknown")
                
                if confidence > confidence_threshold:
                    detection = {
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': [x1, y1, x2, y2],
                        'class_name': class_name
                    }
                    
                    # Categorize based on the CORRECTED class ID
                    # 0 = With_Helmet, 1 = Without_Helmet
                    if class_id == 0:  # With_Helmet
                        with_helmet_detections.append(detection)
                    elif class_id == 1:  # Without_Helmet
                        without_helmet_detections.append(detection)
    
    return with_helmet_detections, without_helmet_detections

def analyze_helmet_status(with_helmet_detections, without_helmet_detections):
    """Analyze detections to determine overall helmet compliance"""
    total_with_helmet = len(with_helmet_detections)
    total_without_helmet = len(without_helmet_detections)
    total_detections = total_with_helmet + total_without_helmet
    
    if total_detections == 0:
        return {
            'status': 'No Detection',
            'message': 'No motorcycle riders detected in the image',
            'confidence': 0.0,
            'is_wearing_helmet': None
        }
    
    # Get the highest confidence detection
    all_detections = with_helmet_detections + without_helmet_detections
    best_detection = max(all_detections, key=lambda x: x['confidence'])
    
    if total_with_helmet > 0 and total_without_helmet == 0:
        # Only helmet detections
        max_confidence = max(det['confidence'] for det in with_helmet_detections)
        return {
            'status': 'Wearing Helmet',
            'message': f'Rider with helmet detected with {max_confidence:.1%} confidence',
            'confidence': max_confidence * 100,
            'is_wearing_helmet': True,
            'details': {
                'with_helmet_count': total_with_helmet,
                'without_helmet_count': 0,
                'total_riders': total_detections
            }
        }
    
    elif total_without_helmet > 0 and total_with_helmet == 0:
        # Only no-helmet detections
        max_confidence = max(det['confidence'] for det in without_helmet_detections)
        return {
            'status': 'Not Wearing Helmet',
            'message': f'Rider without helmet detected with {max_confidence:.1%} confidence',
            'confidence': max_confidence * 100,
            'is_wearing_helmet': False,
            'details': {
                'with_helmet_count': 0,
                'without_helmet_count': total_without_helmet,
                'total_riders': total_detections
            }
        }
    
    elif total_with_helmet > 0 and total_without_helmet > 0:
        # Mixed detections - prioritize safety concern (no helmet)
        max_without_confidence = max(det['confidence'] for det in without_helmet_detections)
        max_with_confidence = max(det['confidence'] for det in with_helmet_detections)
        
        return {
            'status': 'Mixed Detection',
            'message': f'Multiple riders detected: {total_with_helmet} with helmet, {total_without_helmet} without helmet',
            'confidence': max(max_with_confidence, max_without_confidence) * 100,
            'is_wearing_helmet': False,  # False due to safety concern
            'details': {
                'with_helmet_count': total_with_helmet,
                'without_helmet_count': total_without_helmet,
                'total_riders': total_detections,
                'max_with_helmet_confidence': round(max_with_confidence * 100, 2),
                'max_without_helmet_confidence': round(max_without_confidence * 100, 2)
            }
        }
    
    return {
        'status': 'Unknown',
        'message': 'Unable to determine helmet status',
        'confidence': 0.0,
        'is_wearing_helmet': None
    }

@app.on_event("startup")
async def startup_event():
    """Load the model when the API starts"""
    load_model()

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "YOLOv8 Helmet Detection API is running!"}

@app.post("/predict")
async def predict_helmet(file: UploadFile = File(...)):
    """
    Predict whether a person on motorcycle is wearing a helmet using YOLOv8
    """
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read the uploaded image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Convert PIL image to numpy array for YOLOv8
        image_np = np.array(image)
        
        # Make prediction using YOLOv8
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")
        
        # Run inference
        results = model(image_np, conf=0.25)  # confidence threshold
        
        # Process results
        with_helmet_detections, without_helmet_detections = process_yolo_results(results)
        analysis = analyze_helmet_status(with_helmet_detections, without_helmet_detections)
        
        # Prepare response
        result = {
            "filename": file.filename,
            "prediction": analysis['status'],
            "message": analysis['message'],
            "confidence": round(analysis['confidence'], 2),
            "is_wearing_helmet": analysis['is_wearing_helmet'],
            "details": analysis.get('details', {}),
            "raw_detections": {
                "with_helmet_count": len(with_helmet_detections),
                "without_helmet_count": len(without_helmet_detections),
                "total_detections": len(with_helmet_detections) + len(without_helmet_detections),
                "all_detections": [
                    {
                        "class": det['class_name'],
                        "confidence": round(det['confidence'] * 100, 2),
                        "bbox": [round(x, 2) for x in det['bbox']]
                    }
                    for det in with_helmet_detections + without_helmet_detections
                ]
            }
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded YOLOv8 model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        return {
            "model_type": "YOLOv8",
            "model_name": "best_helmet_model.pt",
            "classes": list(CORRECTED_NAMES.values()),
            "class_count": len(CORRECTED_NAMES),
            "status": "loaded",
            "input_size": "640x640 (default)",
            "framework": "Ultralytics"
        }
    except Exception as e:
        return {
            "model_type": "YOLOv8",
            "status": "loaded",
            "error": str(e)
        }

@app.get("/classes")
async def get_model_classes():
    """Get the class names that the model can detect"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "classes": CORRECTED_NAMES,
        "total_classes": len(CORRECTED_NAMES)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)