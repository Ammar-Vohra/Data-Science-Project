from flask import Flask, render_template, request
import os
import numpy as np
import pandas as pd
from src.DataScience.pipeline.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route('/', methods=["GET"])
def homepage():
    # Serves the homepage
    return render_template("index.html")

@app.route("/train", methods=["GET"])
def training():
    try:
        # Executes the training script
        os.system("python main.py")
        return "Training Successful."
    except Exception as e:
        return f"Error during training: {str(e)}"

@app.route("/predict", methods=["POST", "GET"])
def prediction():
    if request.method == "POST":
        try:
            # Collecting input data from the form
            input_data = [
                float(request.form['fixed_acidity']),
                float(request.form['volatile_acidity']),
                float(request.form['citric_acid']),
                float(request.form['residual_sugar']),
                float(request.form['chlorides']),
                float(request.form['free_sulfur_dioxide']),
                float(request.form['total_sulfur_dioxide']),
                float(request.form['density']),
                float(request.form['ph']),
                float(request.form['sulphates']),
                float(request.form['alcohol'])
            ]

            # Converting to DataFrame to match the expected model input
            feature_names = [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
                'pH', 'sulphates', 'alcohol'
            ]
            data = pd.DataFrame([input_data], columns=feature_names)

            # Model prediction
            model = PredictionPipeline()
            prediction = model.predict(data)

            return render_template("results.html", prediction=str(prediction[0]))

        except Exception as e:
            return f"Error during prediction: {str(e)}"

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
