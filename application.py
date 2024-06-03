# application.py
from flask import Flask, request, render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            Income=float(request.form.get('Income')),
            Age=float(request.form.get('Age')),
            Experience=float(request.form.get('Experience')),
            CURRENT_JOB_YRS=float(request.form.get('CURRENT_JOB_YRS')),
            CURRENT_HOUSE_YRS=float(request.form.get('CURRENT_HOUSE_YRS')),
            Married_Single=request.form.get('Married_Single'),
            House_Ownership=request.form.get('House_Ownership'),
            Car_Ownership=request.form.get('Car_Ownership'),
            Profession=request.form.get('Profession'),
            CITY=request.form.get('CITY'),
            STATE=request.form.get('STATE')
        )

        pred_df = data.get_data_as_dataframe()        
        print(pred_df)
        print('Before Prediction')

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        return render_template('index.html', results=results[0], pred_df=pred_df)

if __name__=="__main__":
    app.run(host="0.0.0.0")
