from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from disease_info import disease_info
import os
from datetime import datetime
 
app = Flask(__name__)

# 🔥 LOAD MODEL
model = load_model("model.h5")

# classes
classes = ["Early_Blight", "Healthy", "Late_Blight", "Leaf_Spot"]

# 🔥 NEW: Plant Mapping
plant_map = {
    "Early_Blight": "Tomato",
    "Late_Blight": "Tomato",
    "Leaf_Spot": "Tomato",
    "Healthy": "Tomato"
}

# ensure static folder
if not os.path.exists("static"):
    os.makedirs("static")

# 🔍 Prediction Function
def predict_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    index = np.argmax(prediction)
    confidence = round(np.max(prediction) * 100, 2)

    result = classes[index]
    return result, confidence


@app.route('/', methods=['GET','POST'])
def index():
    if request.method == 'POST':

        if 'file' not in request.files:
            return render_template('index.html', error="No file selected")

        file = request.files['file']

        if file.filename == "":
            return render_template('index.html', error="Please upload an image")

        # unique filename
        filename = datetime.now().strftime("%Y%m%d%H%M%S") + "_" + file.filename
        path = os.path.join("static", filename)
        file.save(path)

        # prediction
        result, confidence = predict_image(path)

        # 🔥 GET PLANT NAME
        plant = plant_map.get(result, "Unknown")

        # language
        lang = request.form.get("lang", "en")

        # disease info
        solution, fertilizer, fert_img = disease_info[result][lang]

        # confidence level
        if confidence > 90:
            level = "High Accuracy ✅"
        elif confidence > 75:
            level = "Moderate ⚠️"
        else:
            level = "Low Confidence ❌"

        # healthy case
        if result == "Healthy":
            solution = "Your plant is healthy 🌱 No disease detected."
            fertilizer = "Maintain proper watering, sunlight, and regular care."

        # impact
        impact = [
            "Instant disease detection",
            "Helps farmers take quick action",
            "Reduces crop loss significantly"
        ]

        # advice
        advice = "Regular monitoring and early treatment can save crops and increase yield."

        return render_template(
            'result.html',
            result=result,
            plant=plant,   # 🔥 SEND PLANT
            solution=solution,
            fertilizer=fertilizer,
            confidence=str(confidence) + "%",
            level=level,
            advice=advice,
            image_path=path,
            lang=lang,
            impact=impact,
            disease_info=disease_info
        )

    return render_template('index.html')


if __name__ == "__main__":
    app.run(debug=True)