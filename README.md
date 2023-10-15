# Tehran House Price Prediction Web Application

Welcome to the Tehran House Price Prediction web application repository! This web application is designed to predict house prices in Tehran, Iran, measured in Rial currency. 

## Project Overview
The Tehran House Price Prediction project involves several key steps to achieve accurate price predictions:

1- Installing and Importing Necessary Libraries: In this step, we install and import the required Python libraries to build and run the web application.

2- Data Loading, Exploration, and Preprocessing: This section is crucial for data preparation. Here, we load the dataset, explore its characteristics, and perform essential preprocessing tasks, such as handling missing values, converting data types, and removing outliers.

3- Modeling with PyCaret: We utilize PyCaret, a low-code ML library, to quickly compare various regression models and get an initial overview of expected model performance.

4- Manual Modeling: In this section, we manually define and train specific regression models, such as Random Forest, XGBoost, CatBoost, and LGBM, giving us more control and flexibility in model selection. Hyperparameter tuning is introduced to optimize model performance.

5- Model Evaluation: Proper model evaluation is critical. Here, we explain how models are assessed using evaluation metrics like Root Mean Square Error (RMSE), Mean Absolute Error (MAE), and R-squared (R2). The results are also stored for future reference.

6- Saving the Model: We save the two selected models, CatBoost and XGBoost, as "CatBoost_model.cbm" and "xgboost_model.pkl," respectively. These models will be used in the web application.

## Files in the Repository
1 - CatBoost_model.cbm and xgboost_model.pkl: These are the trained regression models saved from the notebook and used in the web application.
2- app.py and functions.py: These two files contain the code for our simple web application.
3- preprocessor.pkl: This is the preprocessor used for training, which should also be used in our web application for data preprocessing.
4- requirements.txt: This file lists the libraries used in our virtual environment.
5- tehran-house-price-prediction.ipynb the main code file.



## To see this web application in action, please visit below web page:
https://regressionproblem-abynrvnvghfmcxtxjubz8u.streamlit.app/
