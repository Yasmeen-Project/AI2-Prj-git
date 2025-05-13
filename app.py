from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, jsonify, send_file
from PIL import Image
import torch
from ultralytics import YOLO
import numpy as np
import base64
import io
import cv2
import pickle
from joblib import load as joblib_load

app = Flask(__name__)

# تحميل النماذج

# Helper function to load models gracefully
def load_model_safely(model_path):
    """Attempt to load a model using both pickle and joblib."""
    try:
        print(f"🔄 Attempting to load {model_path} with pickle...")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"✅ Successfully loaded {model_path} with pickle.")
    except (pickle.UnpicklingError, EOFError, AttributeError, ValueError) as e:
        print(f"⚠️ Pickle failed to load {model_path}: {e}")
        print(f"🔄 Attempting to load {model_path} with joblib...")
        try:
            model = joblib_load(model_path)
            print(f"✅ Successfully loaded {model_path} with joblib.")
        except Exception as e:
            print(f"❌ Failed to load {model_path} with joblib: {e}")
            model = None
    return model

# --- Model Loading ---
# Assignment Two
classification_model = load_model("models/classification.keras")
detection_model = YOLO("models/detection.pt")
segmentation_model = YOLO("models/segmentation.pt")

# Assignment One
CNN_no_dropout_model = load_model("models/CNN(withoutDroput)_model.keras")
CNN_with_dropout_model = load_model("models/CNN(withDroput)_model.keras")

# LBP Models
SVM_LBP_model = load_model_safely("models/SVM(LBP)_model.pkl")
KNN_LBP_model = load_model_safely("models/KNN(LBP)_model.pkl")
ANN_LBP_model = load_model("models/ANN_LBP_model.h5")

# ORB Models
SVM_ORB_model = load_model_safely("models/SVM(ORB)_model.pkl")
KNN_ORB_model = load_model_safely("models/KNN(OBB)_model.pkl")
ANN_ORB_model = load_model("models/ANN_ORB_model.h5")

class_names = ["Benign", "Tumor"]

def preprocess_image(image, target_size):
    """معالجة الصورة لتتناسب مع النموذج"""
    img = image.resize(target_size)
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

from skimage.feature import local_binary_pattern
import cv2

# Parameters for LBP
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

# LBP Feature Extraction
def extract_lbp_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, LBP_POINTS + 3),
                             range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize
    return hist.reshape(1, -1)  # Ensure it's in the correct shape

import cv2
import numpy as np

def extract_orb_features(image, target_dim=20):
    """
    Extract ORB features and ensure it matches the target dimensionality.

    Args:
    - image: Image in OpenCV format.
    - target_dim: The expected feature size (default is 20).

    Returns:
    - A numpy array of features with size (1, target_dim).
    """
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=target_dim)
    
    # Detect keypoints and compute descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)
    
    if descriptors is None or len(descriptors) == 0:
        # If no descriptors are found, return a zero vector
        print("⚠️ Warning: No ORB descriptors found. Returning zeros.")
        features = np.zeros((1, target_dim))
    else:
        # Flatten the descriptors
        features = descriptors.flatten()
        
        # If it's larger than target, truncate it
        if len(features) > target_dim:
            features = features[:target_dim]
        # If it's smaller, pad with zeros
        elif len(features) < target_dim:
            features = np.pad(features, (0, target_dim - len(features)), 'constant')
        
        # Reshape to (1, target_dim) for model compatibility
        features = features.reshape(1, -1)
    
    print(f"✅ ORB Features Extracted with Shape: {features.shape}")
    return features


@app.route("/")
def index():
    return render_template('index.html')

@app.route("/process", methods=["POST"])
def process():
    file = request.files["image"]
    model_type = request.form["model_type"]

    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    try:
        img = Image.open(file.stream).convert("RGB")
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # التصنيف
        if model_type in ["classification", "cnn_no_dropout", "cnn_with_dropout"]:
            if model_type == "classification":
                # معالجة لنموذج التصنيف العادي
                img_array = preprocess_image(img, (224, 224))
                model = classification_model
            else:
                # معالجة لنماذج CNN (باستخدام حجم 128x128 كما يبدو من الخطأ)
                img_array = preprocess_image(img, (128, 128))
                model = CNN_no_dropout_model if model_type == "cnn_no_dropout" else CNN_with_dropout_model

            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions[0])
            confidence = float(np.max(predictions[0]))
            result = f"التصنيف: {class_names[predicted_class]} (الثقة: {confidence:.2f})"

            # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)

        # كشف الأجسام
        elif model_type == "detection":
            results = detection_model(img)
            for result_detect in results:
                for box in result_detect.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = box.conf.item()
                    cls = int(box.cls.item())
                    label = class_names[cls]
                    score = round(conf, 3)
                    cv2.rectangle(img_cv, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(img_cv, f"{label} {score}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            result = "تم الكشف عن الكائنات بنجاح!"
            _, img_encoded = cv2.imencode('.jpg', img_cv)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_io.seek(0)
            return send_file(img_io, mimetype='image/jpeg')

        # التقسيم (Segmentation)
        elif model_type == "segmentation":
            results = segmentation_model(img)
            if results and results[0].masks is not None:
                result = "تم الكشف عن الماسكات"
                masks = results[0].masks.data.cpu().numpy()
                for mask in masks:
                    mask = (mask * 255).astype(np.uint8)
                    mask = cv2.resize(mask, (img_cv.shape[1], img_cv.shape[0]))
                    colored_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
                    img_cv = cv2.addWeighted(img_cv, 1, colored_mask, 0.5, 0)
                _, img_encoded = cv2.imencode('.jpg', img_cv)
                img_io = io.BytesIO(img_encoded.tobytes())
                img_io.seek(0)
                return send_file(img_io, mimetype='image/jpeg')
            else:
                result = "لم يتم الكشف عن أي ماسك"
                return render_template("index.html", result=result)

        # --- LBP Models ---
        elif model_type == "svm_lbp":
            features = extract_lbp_features(img_cv)
            model = SVM_LBP_model
            prediction = model.predict(features)
            result = f"التصنيف: {class_names[prediction[0]]}"
            # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)
        
        elif model_type == "knn_lbp":
            features = extract_lbp_features(img_cv)
            model = KNN_LBP_model
            prediction = model.predict(features)
            result = f"التصنيف: {class_names[prediction[0]]}"
            # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)
        
        elif model_type == "ann_lbp":
            features = extract_lbp_features(img_cv)
            model = ANN_LBP_model
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            result = f"التصنيف: {class_names[predicted_class]} (الثقة: {confidence:.2f})"
             # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)

            

        elif model_type == "svm_orb":
            features = extract_orb_features(img_cv, target_dim=20)
            model = SVM_ORB_model
            prediction = model.predict(features)
            result = f"التصنيف: {class_names[prediction[0]]}"
             # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)

        elif model_type == "knn_orb":
            features = extract_orb_features(img_cv, target_dim=20)
            model = KNN_ORB_model
            prediction = model.predict(features)
            result = f"التصنيف: {class_names[prediction[0]]}"
            # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)

        elif model_type == "ann_orb":
            features = extract_orb_features(img_cv, target_dim=20)
            model = ANN_ORB_model
            prediction = model.predict(features)
            predicted_class = np.argmax(prediction[0])
            confidence = float(np.max(prediction[0]))
            result = f"التصنيف: {class_names[predicted_class]} (الثقة: {confidence:.2f})"

            # تجهيز الصورة للإظهار
            img_display = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            img_display = img_display.resize((224, 224))
            img_array_display = np.array(img_display)
            _, img_encoded = cv2.imencode('.jpg', img_array_display)
            img_io = io.BytesIO(img_encoded.tobytes())
            img_base64 = base64.b64encode(img_io.read()).decode('utf-8')
            return render_template("result.html", result=result, image_result=img_base64)

        else:
            return jsonify({"error": "Invalid model type"}), 400
        
        return render_template("result.html", result=result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
