from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        x = float(request.form['x'])
        y = float(request.form['y'])
        z = float(request.form['z'])
        features = np.array([[x, y, z]])
        prediction = model.predict(features)
        result = "Fault Detected" if prediction[0] == 1 else "No Fault Detected"
    except Exception as e:
        result = f"Error: {str(e)}"
    return render_template('Results.html', prediction_text=result)

if __name__ == '__main__':
    app.run(debug=True)


