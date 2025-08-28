from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("new_best_rice_leaf_mobilenetv2.keras")
class_labels = ['Bacterial Leaf Blight', 'Brown Spot', 'Healthy Rice Leaf', 'Leaf Blast','Leaf Scald','Sheath Blight']

# Day-wise treatment dictionary
treatment_info = {
    'Bacterial Leaf Blight': {
        'Day 1': 'Spray copper-based bactericide (e.g., Copper Oxychloride 0.3%).',
        'Day 3': 'Remove infected leaves and improve drainage.',
        'Day 5': 'Apply Streptocycline (0.01%) if symptoms persist.',
        'Day 7': 'Apply potassium fertilizer to boost immunity.'
    },
    'Brown Spot': {
        'Day 1': 'Spray Mancozeb (0.25%) or Carbendazim (0.1%).',
        'Day 3': 'Apply balanced NPK fertilizer.',
        'Day 5': 'Repeat Mancozeb spray.',
        'Day 7': 'Remove severely infected leaves.'
    },
    'Healthy Rice Leaf': {
        'Day 1': 'No treatment needed.',
        'Day 3': 'Maintain hygiene and monitor field.',
        'Day 5': 'Scout for early symptoms.',
        'Day 7': 'Optional: Use neem oil as a preventive.'
    },
    'Leaf Blast': {
        'Day 1': 'Spray Tricyclazole (0.06%) early morning.',
        'Day 3': 'Avoid excess nitrogen. Maintain spacing.',
        'Day 5': 'Spray Isoprothiolane if humidity persists.',
        'Day 7': 'Drain excess water if needed.'
    },
    'Leaf Scald': {
        'Day 1': 'Reduce nitrogen use immediately.',
        'Day 3': 'Spray Carbendazim (0.1%) or Benomyl.',
        'Day 5': 'Improve drainage and remove infected parts.',
        'Day 7': 'Apply another fungicide dose if needed.'
    },
    'Sheath Blight': {
        'Day 1': 'Spray Hexaconazole (0.2%) or Validamycin.',
        'Day 3': 'Remove infected leaves, avoid water stagnation.',
        'Day 5': 'Repeat fungicide if needed.',
        'Day 7': 'Apply potash fertilizer to harden tissues.'
    }
}

# Disease description and reason dictionary
disease_info = {
    'Bacterial Leaf Blight': 'Caused by *Xanthomonas oryzae*. Occurs due to high humidity, stagnant water, and infected seeds.',
    'Brown Spot': 'Caused by *Bipolaris oryzae*. Triggered by poor soil nutrition, water stress, and high temperatures.',
    'Healthy Rice Leaf': 'The leaf is free from diseases. Maintain field hygiene to ensure continued health.',
    'Leaf Blast': 'Caused by *Magnaporthe oryzae*. Spread by wind, rain splash, and dense planting under high humidity.',
    'Leaf Scald': 'Often due to potassium deficiency or fungal infection under hot, wet conditions.',
    'Sheath Blight': 'Caused by *Rhizoctonia solani*. Promoted by dense planting, high nitrogen use, and waterlogged conditions.'
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
            img = Image.open(file).convert("RGB")
            img = img.resize((224, 224))
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

    return render_template("index.html", prediction=prediction, img_path=img_path, treatment_plan=treatment_plan, description=description)


if __name__ == "__main__":
    app.run(debug=True)
