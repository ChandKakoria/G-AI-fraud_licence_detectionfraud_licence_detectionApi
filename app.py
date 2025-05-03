from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io

# Initialize Flask app
app = Flask(__name__)

# Load your trained model
model = load_model('license_cnn_final.h5')

# Prediction function
def predict_image_from_file(file_stream):
    img = image.load_img(file_stream, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    prediction = model.predict(img_array)[0][0]
    is_fraud = prediction < 0.5

    return {
        "prediction": "fraud" if is_fraud else "genuine",
        "probability_genuine": float(prediction),
        "probability_fraud": float(1 - prediction)
    }

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Use file stream directly, no need to save to disk
        result = predict_image_from_file(io.BytesIO(file.read()))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)

