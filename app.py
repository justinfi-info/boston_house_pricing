import os
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np

app = Flask(__name__)

# load the model
regmodel = pickle.load(open('regmodel.pkl', 'rb'))

# optional scaler (if present)
scalar = None
if os.path.exists('scalar.pkl'):
    scalar = pickle.load(open('scalar.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json('data')
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify(output[0])

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract form data
        form_data = request.form
        data = {key: float(value) for key, value in form_data.items()}
        values = np.array(list(data.values())).reshape(1, -1)

        # Apply scaler if available
        if scalar is not None:
            values = scalar.transform(values)

        # Make prediction
        prediction = regmodel.predict(values)[0]

        # Render template with prediction
        return render_template('home.html', prediction_text=f'Predicted Price: ${prediction:.2f}')
    except Exception as e:
        return render_template('home.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
