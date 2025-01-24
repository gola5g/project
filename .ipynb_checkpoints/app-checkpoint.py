from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('fraud_detection_model.pkl')

# Define route for home (root)
@app.route('/')
def home():
    return "Credit Card Fraud Detection API is Running!"

# Define route for fraud prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from the request (JSON format)
    data = request.get_json()
    
    # Convert JSON data into a DataFrame for processing
    input_data = pd.DataFrame(data, index=[0])

    # Preprocess the 'Amount' column by scaling it (if included in the input)
    if 'Amount' in input_data.columns:
        scaler = StandardScaler()
        input_data['Normalized_Amount'] = scaler.fit_transform(input_data['Amount'].values.reshape(-1, 1))
        input_data = input_data.drop(['Amount'], axis=1)  # Drop original 'Amount' column

    # Drop 'Class', 'Time' columns if they exist (as they're not used for prediction)
    input_data = input_data.drop(['Class', 'Time'], axis=1, errors='ignore')

    # Make predictions using the model
    prediction = model.predict(input_data)

    # Return prediction result in JSON format
    return jsonify({'fraud': bool(prediction[0])})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
