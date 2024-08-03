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
        data = CustomData(
            Year_Birth=int(request.form.get('Year_Birth')),
            Dt_Customer=str(request.form.get('Dt_Customer')),
            Education=request.form.get('Education'),
            Income=int(request.form.get('Income')),
            Kidhome=int(request.form.get('Kidhome')),
            Teenhome=int(request.form.get('Teenhome')),
            Recency=int(request.form.get('Recency')),
            MntWines=int(request.form.get('MntWines')),
            MntFruits=int(request.form.get('MntFruits')),
            MntMeatProducts=int(request.form.get('MntMeatProducts')),
            MntFishProducts=int(request.form.get('MntFishProducts')),
            MntSweetProducts=int(request.form.get('MntSweetProducts')),
            MntGoldProds=int(request.form.get('MntGoldProds')),
            NumDealsPurchases=int(request.form.get('NumDealsPurchases')),
            NumWebPurchases=int(request.form.get('NumWebPurchases')),
            NumCatalogPurchases=int(request.form.get('NumCatalogPurchases')),
            NumStorePurchases=int(request.form.get('NumStorePurchases')),
            NumWebVisitsMonth=int(request.form.get('NumWebVisitsMonth')),
            Complain=int(request.form.get('Complain')),
            Marital_Status=request.form.get('Marital_Status')

        )
        pred_df = data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline = PredictPipeline()
        print("Mid Prediction")
        results = predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html', results=results[0])

if __name__ == "__main__":
    app.run(host="0.0.0.0")