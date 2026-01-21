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
import os
import json
import uuid
import logging
from datetime import datetime

import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

# ================= CONFIG =================
PORT = int(os.environ.get("PORT", 8080))
UPLOAD_DIR = "uploadimages"
MODEL_PATH = "models/plant_disease_recog_model_pwp.keras"
DATA_PATH = "plant_disease.json"

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ================= LOGGING =================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PHASAL")

# ================= APP =================
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": [
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "https://phasalaipr.onrender.com"
        ]
    }
})

# ================= GLOBALS (LAZY LOAD) =================
model = None
plant_disease = {}

# ================= HELPERS =================
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    if not file or file.filename == "":
        return False, "No file selected"

    if not allowed_file(file.filename):
        return False, "Only JPG / PNG allowed"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size > MAX_FILE_SIZE:
        return False, "File too large (max 10MB)"

    return True, "Valid"

def load_model_once():
    global model
    if model is None:
        logger.info("🔄 Loading TensorFlow model...")
        model = tf.keras.models.load_model(MODEL_PATH)
        logger.info("✅ Model loaded successfully")
    return model

def load_disease_data():
    global plant_disease
    if not plant_disease:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            plant_disease = json.load(f)

def preprocess_image(image_path):
    # ⚠️ FIXED SIZE (do NOT use model.input_shape on Render)
    image = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    image_array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(image_array, axis=0).astype(np.float32)

# ================= ROUTES =================

@app.get("/api/health")
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }), 200

@app.post("/api/predict")
def api_predict():
    try:
        if "img" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["img"]
        ok, msg = validate_image_file(file)
        if not ok:
            return jsonify({"error": msg}), 400

        ext = os.path.splitext(file.filename)[1].lower()
        filename = secure_filename(f"leaf_{uuid.uuid4().hex}{ext}")
        save_path = os.path.join(UPLOAD_DIR, filename)
        file.save(save_path)

        logger.info(f"📸 Image saved: {filename}")

        # Lazy load (VERY IMPORTANT for Render)
        load_disease_data()
        mdl = load_model_once()

        preds = mdl.predict(preprocess_image(save_path), verbose=0)[0]
        class_idx = int(np.argmax(preds))
        confidence = float(preds[class_idx])

        if isinstance(plant_disease, list):
            disease_info = plant_disease[class_idx] if class_idx < len(plant_disease) else {}
        else:
            disease_info = plant_disease.get(str(class_idx), {})

        disease_info.setdefault("name", f"Class {class_idx}")
        disease_info.setdefault("cause", "Information not available")
        disease_info.setdefault("cure", "Information not available")

        return jsonify({
            "result": True,
            "classIndex": class_idx,
            "confidence": confidence,
            "prediction": disease_info,
            "probs": preds.tolist(),
            "imageUrl": f"{request.host_url.rstrip('/')}/uploadimages/{filename}",
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        logger.error("❌ Prediction failed", exc_info=True)
        return jsonify({"error": "Prediction failed"}), 500

@app.get("/uploadimages/<path:filename>")
def uploaded_images(filename):
    return send_from_directory(UPLOAD_DIR, filename)

@app.get("/api/stats")
def stats():
    return jsonify({
        "uploads_stored": len(os.listdir(UPLOAD_DIR)),
        "total_classes": len(plant_disease),
        "model_ready": model is not None,
        "version": "1.0.0"
    })

@app.get("/")
def root():
    return jsonify({
        "message": "PHASAL - Plant Disease Detection API",
        "endpoints": {
            "health": "/api/health",
            "predict": "/api/predict (POST)",
            "stats": "/api/stats",
            "images": "/uploadimages/<filename>"
        }
    })

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
