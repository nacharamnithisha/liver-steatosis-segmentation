import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)

# ---------------- CONFIG ----------------
MODEL_PATH = "model/unet_model.h5"
UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
IMG_SIZE = 256

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load trained model
model = load_model(MODEL_PATH)

# ---------------- ROUTE ----------------
@app.route("/", methods=["GET", "POST"])
def index():
    original_image = None
    output_image = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename != "":
            original_image = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(original_image)

            # Read grayscale image
            img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape

            # Model preprocessing
            img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img_norm = img_resized / 255.0
            img_input = img_norm.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            # Liver segmentation
            pred = model.predict(img_input)[0, :, :, 0]
            liver_mask = cv2.resize(pred, (w, h)) > 0.4

            # ---------- STEATOSIS LOGIC (FIXED) ----------
            liver_pixels = img[liver_mask]

            if len(liver_pixels) > 0:
                mean_intensity = np.mean(liver_pixels)
            else:
                mean_intensity = np.mean(img)

            # Steatosis = brighter than liver average
            fatty_region = (img > mean_intensity + 15) & liver_mask

            # Remove noise
            fatty_region = cv2.medianBlur(fatty_region.astype(np.uint8)*255, 5) > 0

            # Overlay
            overlay = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            overlay[fatty_region] = [0, 0, 255]

            # Text
            cv2.putText(
                overlay,
                "Segmented Liver Region (Steatosis Highlighted)",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            output_image = os.path.join(OUTPUT_FOLDER, "result.png")
            cv2.imwrite(output_image, overlay)

    return render_template(
        "index.html",
        original_image=original_image,
        output_image=output_image
    )

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

