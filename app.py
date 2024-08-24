from flask import Flask, request, render_template

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


## Route for a home page

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomData(
                person_age=int(request.form.get('person_age')),
                person_income=float(request.form.get('person_income')),
                person_home_ownership=request.form.get('person_home_ownership'),
                person_emp_length=float(request.form.get('person_emp_length')),
                loan_intent=request.form.get('loan_intent'),
                loan_amnt=float(request.form.get('loan_amnt')),
                loan_int_rate=float(request.form.get('loan_int_rate')),
                cb_person_default_on_file=request.form.get('cb_person_default_on_file'),
                cb_person_cred_hist_length=float(request.form.get('cb_person_cred_hist_length')),
                loan_percent_income=float(request.form.get('loan_percent_income'))
            )
            pred_df = data.get_data_as_data_frame()
            print(pred_df)
            print("Before Prediction")

            predict_pipeline = PredictPipeline()
            print("Mid Prediction")
            results = predict_pipeline.predict(pred_df)
            print(results)
            print("After Prediction")
            return render_template('home.html', results=results[0])

        except Exception as e:
            print(f"Error occurred: {e}")
            return render_template('home.html', error=str(e))


if __name__ == "__main__":
    app.run(host="0.0.0.0")