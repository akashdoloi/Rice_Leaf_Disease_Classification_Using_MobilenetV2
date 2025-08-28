from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)

# Load trained model and class labels
MODEL_PATH = "rice_leaf_mobilenetv2.keras"
LABELS_PATH = "rice_labels.txt"

model = load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    class_labels = [line.strip() for line in f.readlines()]

# Disease descriptions
disease_info = {
    'Bacterial Leaf Blight': 'Caused by Xanthomonas oryzae. High humidity and stagnant water promote it.',
    'Brown Spot': 'Caused by Bipolaris oryzae. Triggered by poor nutrition and water stress.',
    'Healthy Rice Leaf': 'Leaf is healthy. Maintain hygiene and proper monitoring.',
    'Leaf Blast': 'Caused by Magnaporthe oryzae. Spread by wind, rain, and dense planting.',
    'Leaf Scald': 'Due to potassium deficiency or fungal infection in wet conditions.',
    'Sheath Blight': 'Caused by Rhizoctonia solani. Promoted by dense planting and high nitrogen.'
}

# Treatments
treatment_info = {
    'Bacterial Leaf Blight': {'Day 1': 'Spray copper bactericide', 'Day 3': 'Remove infected leaves', 'Day 5': 'Apply Streptocycline', 'Day 7': 'Apply potassium'},
    'Brown Spot': {'Day 1': 'Spray Mancozeb', 'Day 3': 'Apply balanced NPK', 'Day 5': 'Repeat spray', 'Day 7': 'Remove infected leaves'},
    'Healthy Rice Leaf': {'Day 1': 'No treatment', 'Day 3': 'Maintain hygiene', 'Day 5': 'Scout field', 'Day 7': 'Optional preventive neem spray'},
    'Leaf Blast': {'Day 1': 'Spray Tricyclazole', 'Day 3': 'Reduce nitrogen', 'Day 5': 'Spray Isoprothiolane', 'Day 7': 'Drain excess water'},
    'Leaf Scald': {'Day 1': 'Reduce nitrogen', 'Day 3': 'Spray Carbendazim', 'Day 5': 'Improve drainage', 'Day 7': 'Repeat fungicide if needed'},
    'Sheath Blight': {'Day 1': 'Spray Hexaconazole', 'Day 3': 'Remove infected leaves', 'Day 5': 'Repeat spray', 'Day 7': 'Apply potash fertilizer'}
}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_path = None
    treatment_plan = None
    description = None

    if request.method == "POST":
        file = request.files["file"]
        if file:
            img = Image.open(file).convert("RGB").resize((224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            predicted_class = class_labels[np.argmax(preds)]
            confidence = round(np.max(preds) * 100, 2)

            prediction = f"{predicted_class} ({confidence}%)"
            description = disease_info.get(predicted_class, "No information available.")
            treatment_plan = treatment_info.get(predicted_class, {})

            img_path = os.path.join("static", "preview.jpg")
            img.save(img_path)

    return render_template(
        "index.html",
        prediction=prediction,
        img_path=img_path,
        treatment_plan=treatment_plan,
        description=description
    )

if __name__ == "__main__":
    app.run(debug=True)
