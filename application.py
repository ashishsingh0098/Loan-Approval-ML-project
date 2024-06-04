import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import StandardScaler

from src.pipeline.predict_pipeline import PredictPipeline
from src.pipeline.predict_pipeline import CustomData

application = Flask(__name__)

app = application

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Income=int(request.form.get('Income')),
            Age=int(request.form.get('Age')),
            Experience=int(request.form.get('Experience')),
            Married_Single=request.form.get('Married_Single'),
            House_Ownership=request.form.get('House_Ownership'),
            Car_Ownership=request.form.get('Car_Ownership'),
            Profession=request.form.get('Profession'),
            CITY=request.form.get('CITY'),
            STATE=request.form.get('STATE'),
            CURRENT_JOB_YRS=int(request.form.get('CURRENT_JOB_YRS')),
            CURRENT_HOUSE_YRS=int(request.form.get('CURRENT_HOUSE_YRS'))
        )

        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print('Before Prediction')

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")
