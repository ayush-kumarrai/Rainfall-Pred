from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load('rainfall_model.pkl')
scaler = joblib.load('scaler-2.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        latitude = data['Latitude']
        longitude = data['Longitude']
        
        if latitude is None or longitude is None :
            return jsonify({"error": "Invalid data"}), 400

        input_data = np.array([[latitude, longitude]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)
        
        result = prediction[0]
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
