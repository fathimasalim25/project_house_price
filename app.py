from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model and the fitted scaler
model_path = 'model_new (2).pkl'  # Path to your trained model
scaler_path = 'scaler (5).pkl'  # Path to your fitted scaler

# Load model
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

# Load the fitted scaler
with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Route for the home page (input form)
@app.route('/')
def home():
    return render_template('index.html')  # Render the HTML form

# Route to handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        long = float(request.form['long'])
        lat = float(request.form['lat'])
        sqft_living = float(request.form['sqft_living'])
        sqft_lot = float(request.form['sqft_lot'])
        sqft_above = float(request.form['sqft_above'])

        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'long': [long],
            'lat': [lat],
            'sqft_living': [sqft_living],
            'sqft_lot': [sqft_lot],
            'sqft_above': [sqft_above],
        })

        # Scale the input data using the fitted scaler
        input_data_scaled = scaler.transform(input_data)

        # Predict the price using the model
        predicted_price = model.predict(input_data_scaled)

        # Convert predicted price to a string format for rendering
        predicted_price = round(predicted_price[0], 2)

        # Render the result page with the predicted price
        return render_template('result.html', price=predicted_price)

    except KeyError as e:
        return f"Missing input: {str(e)}", 400
    except ValueError as e:
        return f"Invalid input value: {str(e)}", 400
    except Exception as e:
        # Generic error handling for the prediction step
        return f"An error occurred during prediction: {str(e)}", 500

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

