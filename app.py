from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        area = float(request.form['area'])  # GrLivArea
        bedrooms = int(request.form['bedrooms'])  # BedroomAbvGr
        bathrooms = int(request.form['bathrooms'])  # FullBath

        input_data = np.array([[area, bedrooms, bathrooms]])
        prediction = model.predict(input_data)[0]

        return render_template('index.html',
                               prediction_text=f"Estimated House Price: ₹{prediction:,.2f}")
    except ValueError:
        return render_template('index.html', prediction_text="⚠ Please enter valid numbers.")

if __name__ == '__main__':
    app.run(debug=True)