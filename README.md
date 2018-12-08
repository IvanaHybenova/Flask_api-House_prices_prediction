# Flask_api-House_prices_prediction
Predicts house prices

Data: 
Mock-up dataset from the internet (1883 instances, 26 features)

Goal: 
Predict the fair transaction price of a property before it's sold.

Challenge: 
Dirty data(fixing structural errors, dropping wrong instances, flagging missing categorical data), feature engineering

Algorithms: Lasso, Ridge, Elastic Net, Gradient boosting regressor, Random forest regressor

Measures: Mean absolute error, adjusted R squared

Project delivery: Python script executing locally hosted flask api, that takes in raw data, preprocess them, do the predictions and provide downloadable zipped .xlsx file with provided dataset alongside with column of predicted price
