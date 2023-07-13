from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('bestt_model.pkl', 'rb'))

# Define the Flask application
# Define the Flask application
app = Flask(__name__)

# Define the route for the index page
@app.route('/')
def index():
    return render_template('index.html')

# Define the route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the form submission
        calories = float(request.form['calories'])
        dream_weight = float(request.form['dream_weight'])
        actual_weight = float(request.form['actual_weight'])
        age = float(request.form['age'])
        duration = float(request.form['duration'])
        heart_rate = float(request.form['heart_rate'])
        height = float(request.form['height'])
        gender = request.form['gender']
        weather_conditions = request.form['weather_conditions']

        # Calculate BMI
        bmi = actual_weight / (height ** 2)

        # Create a DataFrame with user input
        user_input = pd.DataFrame(
            [[calories, dream_weight, actual_weight, age, duration, heart_rate, bmi, gender, weather_conditions]],
            columns=['Calories', 'Dream Weight', 'Actual Weight', 'Age', 'Duration', 'Heart Rate', 'BMI', 'Gender',
                     'Weather Conditions'])

        # Encode categorical variables
        user_input['Gender'] = user_input['Gender'].apply(lambda x: 1 if x == 'Female' else 0)
        user_input['Weather Conditions'] = user_input['Weather Conditions'].map({'Sunny': 1, 'Cloudy': 2, 'Rainy': 0})

        # Ensure correct number of features
        expected_features = model.feature_importances_
        missing_features = set(expected_features) - set(user_input.columns)
        for feature in missing_features:
            user_input[feature] = 0

        # Reorder columns to match the training data
        user_input = user_input[expected_features]

        # Make a prediction using the model
        prediction = int(round(model.predict(user_input)[0]))

        # Define intensity descriptions
        intensity_descriptions = {
            1: 'Very low intensity: Suitable for individuals with minimal physical activity, such as walking at a leisurely pace, stretching or yoga, and gentle swimming. However, please note that your preference is prioritized.',
            2: 'Low intensity: Suitable for beginners or individuals with limited physical activity, such as light jogging or running, cycling at a relaxed pace, and beginner\'s aerobics. However, please remember that your preference takes priority.',
            3: 'Moderate intensity: Suitable for individuals who engage in regular physical activity, including brisk walking, dancing, and water aerobics. However, please keep in mind that your preference is given priority.',
            4: 'Medium intensity: Suitable for individuals with moderate fitness and physical activity, such as power walking, cycling at a moderate pace, and Zumba. However, please remember that your preference is given priority.',
            5: 'Moderate to high intensity: Suitable for individuals with moderate to high fitness levels, including jogging or running at a moderate pace, high-intensity interval training (HIIT), and kickboxing. However, please note that your preference is given priority.',
            6: 'High intensity: Suitable for individuals with a high level of fitness and physical activity, such as running at a fast pace, circuit training, and CrossFit. However, please keep in mind that your preference is given priority.',
            7: 'High intensity: Suitable for individuals with a high level of fitness and physical activity, such as advanced HIIT workouts, competitive sports (e.g., soccer, basketball), and spinning or indoor cycling classes. However, please remember that your preference is given priority.',
            8: 'Very high intensity: Suitable for individuals with very high fitness and physical activity, including sprinting or interval sprints, plyometric exercises, and heavy weightlifting. However, please keep in mind that your preference is given priority.',
            9: 'Very higher intensity: Suitable for individuals with very high fitness and physical activity, such as advanced CrossFit workouts, box jumps, and Olympic weightlifting. However, please note that your preference is given priority.',
            10: 'Extremely high intensity: Suitable for athletes or individuals with exceptional fitness levels, including professional sports training, marathon running, and elite-level strength and conditioning programs. However, please remember that your preference is given priority.'
        }

        # Retrieve the description based on the predicted intensity level
        description = intensity_descriptions.get(prediction, "Unknown")

        # Prepare the response as a JSON object
        response = {
            'intensity': prediction,
            'description': description
        }

        return render_template('index.html', prediction=response)

    except Exception as e:
        return jsonify({'error': str(e)})
# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)