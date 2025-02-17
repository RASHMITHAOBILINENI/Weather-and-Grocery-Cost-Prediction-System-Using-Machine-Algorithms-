# app/app.py
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

weather_model = joblib.load('../models/weather_model.joblib')
grocery_model = joblib.load('../models/grocery_model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input
    location = request.form['location']
    item = request.form['item']
    
    # Fetch real-time weather data (pseudo-code)
    weather_data = get_weather_from_api(location)  
    
   
    input_df = pd.DataFrame([{
        'temp_avg': (weather_data['temp_min'] + weather_data['temp_max']) / 2,
        'humidity': weather_data['humidity'],
        'precipitation': weather_data['precipitation'],
        **weather_data['weather_descriptions']  
    }])
    
    # Predict price
    predicted_price = grocery_model.predict(input_df)[0]
    
    return render_template('index.html', 
                         prediction=f"Predicted {item} price: ₹{predicted_price:.2f}")

def get_weather_from_api(location):
    # API response
    return {
        'temp_min': 18,
        'temp_max': 28,
        'humidity': 65,
        'precipitation': 0.2,
        'weather_descriptions': {'weather_description_rain': 1}
    }

if __name__ == '__main__':
    app.run(debug=True)