# from flask import Flask, render_template,request,redirect,send_from_directory,url_for
# import numpy as np
# import json
# import uuid
# import tensorflow as tf

# app = Flask(__name__)
# model = tf.keras.models.load_model("models/plant_disease (1).keras")
# label = ['Apple___Apple_scab',
#  'Apple___Black_rot',
#  'Apple___Cedar_apple_rust',
#  'Apple___healthy',
#  'Background_without_leaves',
#  'Blueberry___healthy',
#  'Cherry___Powdery_mildew',
#  'Cherry___healthy',
#  'Corn___Cercospora_leaf_spot Gray_leaf_spot',
#  'Corn___Common_rust',
#  'Corn___Northern_Leaf_Blight',
#  'Corn___healthy',
#  'Grape___Black_rot',
#  'Grape___Esca_(Black_Measles)',
#  'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
#  'Grape___healthy',
#  'Orange___Haunglongbing_(Citrus_greening)',
#  'Peach___Bacterial_spot',
#  'Peach___healthy',
#  'Pepper,_bell___Bacterial_spot',
#  'Pepper,_bell___healthy',
#  'Potato___Early_blight',
#  'Potato___Late_blight',
#  'Potato___healthy',
#  'Raspberry___healthy',
#  'Soybean___healthy',
#  'Squash___Powdery_mildew',
#  'Strawberry___Leaf_scorch',
#  'Strawberry___healthy',
#  'Tomato___Bacterial_spot',
#  'Tomato___Early_blight',
#  'Tomato___Late_blight',
#  'Tomato___Leaf_Mold',
#  'Tomato___Septoria_leaf_spot',
#  'Tomato___Spider_mites Two-spotted_spider_mite',
#  'Tomato___Target_Spot',
#  'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
#  'Tomato___Tomato_mosaic_virus',
#  'Tomato___healthy']

# with open("plant_disease.json",'r') as file:
#     plant_disease = json.load(file)

# # print(plant_disease[4])

# @app.route('/uploadimages/<path:filename>')
# def uploaded_images(filename):
#     return send_from_directory('./uploadimages', filename)

# @app.route('/',methods = ['GET'])
# def home():
#     return render_template('home.html')

# def extract_features(image):
#     image = tf.keras.utils.load_img(image,target_size=(160,160))
#     feature = tf.keras.utils.img_to_array(image)
#     feature = np.array([feature])
#     return feature

# def model_predict(image):
#     img = extract_features(image)
#     prediction = model.predict(img)
#     # print(prediction)
#     prediction_label = plant_disease[prediction.argmax()]
#     return prediction_label

# @app.route('/upload/',methods = ['POST','GET'])
# def uploadimage():
#     if request.method == "POST":
#         image = request.files['img']
#         temp_name = f"uploadimages/temp_{uuid.uuid4().hex}"
#         image.save(f'{temp_name}_{image.filename}')
#         print(f'{temp_name}_{image.filename}')
#         prediction = model_predict(f'./{temp_name}_{image.filename}')
#         return render_template('home.html',result=True,imagepath = f'/{temp_name}_{image.filename}', prediction = prediction )
    
#     else:
#         return redirect('/')
        
    
# if __name__ == '__main__':
#     app.run(debug=True, port=8080)  # Ya koi free port
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import numpy as np
import json
import uuid
import os
import tensorflow as tf
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# CORS Configuration
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173","https://phasalaipr.onrender.com"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Load Model
try:
    model = tf.keras.models.load_model("models/plant_disease_recog_model_pwp.keras")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Model loading failed: {e}")
    model = None

# Load Disease Data
try:
    with open("plant_disease.json", "r", encoding="utf-8") as f:
        plant_disease = json.load(f)
    logger.info(f"✅ Disease data loaded: {len(plant_disease)} classes")
except Exception as e:
    logger.error(f"❌ Disease data loading failed: {e}")
    plant_disease = {}

# Upload Directory
UPLOAD_DIR = os.path.join(os.getcwd(), "uploadimages")
os.makedirs(UPLOAD_DIR, exist_ok=True)
logger.info(f"✅ Upload directory: {UPLOAD_DIR}")

# Config
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Error Handler
@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Error: {str(e)}", exc_info=True)
    return jsonify({"error": "Internal server error", "details": str(e)}), 500

# Health Check
@app.get("/api/health")
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "classes": len(plant_disease),
        "timestamp": datetime.now().isoformat()
    }), 200

# Serve Uploaded Images
@app.get("/uploadimages/<path:filename>")
def uploaded_images(filename):
    """Serve uploaded leaf images"""
    try:
        return send_from_directory(UPLOAD_DIR, filename)
    except Exception as e:
        return jsonify({"error": "File not found"}), 404

# Validation
def allowed_file(filename):
    """Check if file extension is allowed"""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validate uploaded image"""
    if not file or file.filename == "":
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "Only JPG/PNG files allowed"
    
    if len(file.read()) > MAX_FILE_SIZE:
        file.seek(0)
        return False, "File too large (max 10MB)"
    
    file.seek(0)
    return True, "Valid"

# Image Processing
def preprocess_image(image_path: str):
    """Load and preprocess image for model"""
    try:
        # Get model input shape
        input_shape = model.input_shape
        target_size = (input_shape[1], input_shape[2])
        
        # Load image
        image = tf.keras.utils.load_img(image_path, target_size=target_size)
        image_array = tf.keras.utils.img_to_array(image)
        image_batch = np.expand_dims(image_array, axis=0).astype(np.float32)
        
        # Normalize if needed (check if model expects normalized input)
        # Uncomment if model was trained on normalized data (0-1 range)
        # image_batch = image_batch / 255.0
        
        return image_batch
    except Exception as e:
        logger.error(f"Image preprocessing error: {e}")
        raise

# Prediction
def get_prediction(image_path: str):
    """Get model prediction"""
    if not model:
        raise Exception("Model not loaded")
    
    try:
        # Preprocess
        processed_image = preprocess_image(image_path)
        
        # Predict
        predictions = model.predict(processed_image, verbose=0)
        probs = predictions[0].astype(float)
        
        # Get class index
        class_idx = int(np.argmax(probs))
        confidence = float(probs[class_idx])
        
        # Get top 3 predictions
        top_3_idx = np.argsort(probs)[-3:][::-1].tolist()
        
        # Get disease info
        disease_info = plant_disease[class_idx] if isinstance(plant_disease, list) else plant_disease.get(str(class_idx), plant_disease.get(class_idx, {}))
        
        # Handle string responses
        if isinstance(disease_info, str):
            disease_info = {"name": disease_info, "cause": "Unknown", "cure": "Unknown"}
        
        # Ensure all fields exist
        disease_info.setdefault("name", f"Class {class_idx}")
        disease_info.setdefault("cause", "Information not available")
        disease_info.setdefault("cure", "Information not available")
        
        return {
            "class_idx": class_idx,
            "confidence": confidence,
            "disease_info": disease_info,
            "probabilities": probs.tolist(),
            "top_3": top_3_idx
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

# Main Prediction Endpoint
@app.post("/api/predict")
def api_predict():
    """Main prediction endpoint"""
    try:
        # Validate request
        if "img" not in request.files:
            return jsonify({"error": "No image file provided"}), 400
        
        file = request.files["img"]
        
        # Validate file
        is_valid, message = validate_image_file(file)
        if not is_valid:
            return jsonify({"error": message}), 400
        
        # Save file
        ext = os.path.splitext(file.filename)[1].lower()
        filename = secure_filename(f"leaf_{uuid.uuid4().hex}{ext}")
        save_path = os.path.join(UPLOAD_DIR, filename)
        
        try:
            file.save(save_path)
            logger.info(f"File saved: {filename}")
        except Exception as e:
            logger.error(f"File save error: {e}")
            return jsonify({"error": "Failed to save image"}), 500
        
        # Get prediction
        try:
            pred_result = get_prediction(save_path)
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            return jsonify({"error": "Prediction failed", "details": str(e)}), 500
        
        # Build image URL
        base_url = request.host_url.rstrip("/")
        image_url = f"{base_url}/uploadimages/{filename}"
        
        # Build response
        response = {
            "result": True,
            "classIndex": pred_result["class_idx"],
            "prediction": pred_result["disease_info"],
            "confidence": pred_result["confidence"],
            "imageUrl": image_url,
            "probs": pred_result["probabilities"],
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Prediction success: Class {pred_result['class_idx']} ({pred_result['confidence']:.2%})")
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

# Stats Endpoint (optional)
@app.get("/api/stats")
def stats():
    """Get app statistics"""
    try:
        uploaded_count = len(os.listdir(UPLOAD_DIR)) if os.path.exists(UPLOAD_DIR) else 0
        return jsonify({
            "total_classes": len(plant_disease),
            "uploads_stored": uploaded_count,
            "model_ready": model is not None,
            "version": "1.0.0"
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Root Endpoint
@app.get("/")
def root():
    """Root endpoint"""
    return jsonify({
        "message": "PHASAL - Plant Disease Detection API",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)",
            "stats": "/api/stats",
            "images": "/uploadimages/<filename>"
        }
    }), 200

# 404 Handler
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

# 405 Handler
@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed"}), 405

# Production Run
if __name__ == "__main__":
    # Development
    app.run(debug=True, host="0.0.0.0", port=8080)
    
    # Production (uncomment and use gunicorn instead)
    # from waitress import serve
    # serve(app, host='0.0.0.0', port=8080)


