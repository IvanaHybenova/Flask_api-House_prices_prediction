# Flask_api-House_prices_prediction
Predicts house prices

Data: 
Mock-up dataset from the internet (1883 instances, 26 features)

Goal: 
Predict the fair transaction price of a property before it's sold.

Challenge: 
Dirty data(fixing structural errors, dropping wrong instances, flagging missing categorical data), feature engineering

Algorithms: 
Lasso, Ridge, Elastic Net, Gradient boosting regressor, Random forest regressor

Measures: 
Mean absolute error, adjusted R squared

Project delivery: 
Python script executing locally hosted flask api, that takes in raw data, preprocess them, do the predictions and provide downloadable zipped .xlsx file with provided dataset alongside with column of predicted price

Files: 
Real-Estate_Tycoon.py - Python script that contains exploration of the data, data cleaning and modeling

flask_predict_api.py - Python scirpt with the application

real_estate_data.csv - Dataset provided for the project

raw_data.csv - This is actually hold out set (test.set) after the split I saved for being able to test the app

Instructions: Download raw_unseen_data.csv, zip file with the model (make sure to have final_model.pkl in separate "model" folder created among the under downloaded files) and flask_predict_api.py.

Through your command line navigate to the folder you are storing these files. Make sure you have python path in your enviroment variables and run command python flask_predict_api.py

From your browser navigate to http://localhost:8000/apidocs. Click on predict_api and then try it out!. Insert raw_data.csv and press execute. After some time scroll down and click on Download the zip.file, which contains the predictions.
