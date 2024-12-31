import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime, timedelta


from airflow import DAG
from airflow.operators.python import PythonOperator


def extraction():
    try:
        df = pd.read_csv("/opt/airflow/dags/new_data/flights.csv")
        df.dropna(inplace = True)
        df.drop_duplicates(inplace = True)
        df.to_csv("/opt/airflow/dags/new_data/extracted_data.csv", index = False)
        print("The data is extracted successfully")
    except Exception as e:
        print(f"There is a problem in extraction of data : {e}")

def transformation():
    try:
        df = pd.read_csv("/opt/airflow/dags/new_data/extracted_data.csv")
        df = df.drop(columns = ['travelCode',  'userCode', 'time', 'distance', 'date'])
        df = pd.get_dummies(df, columns = ['from', 'to', 'flightType', 'agency'], drop_first = True)
        df.to_pickle( "/opt/airflow/dags/new_data/transformed_data.pkl")
       
        print("The Transformed data has saved successfully")
    except Exception as e:
        print(f"There is error in Transforming the data: {e}")

def training():
    try:
        df = joblib.load("/opt/airflow/dags/new_data/transformed_data.pkl")

        X = df.drop(columns = ['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

        rf = RandomForestRegressor()
        rf.fit(X_train, y_train)
        joblib.dump(rf,"/opt/airflow/dags/new_data/trained_model.pkl")
        print("The model has been trained and saved successfully")
    except Exception as e:
        print(f"There is some error while training: {e}")

def evaluation():
    try:
        df = joblib.load("/opt/airflow/dags/new_data/transformed_data.pkl")

        X = df.drop(columns = ['price'])
        y = df['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

        model = joblib.load("/opt/airflow/dags/new_data/trained_model.pkl")
        y_predict = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_predict)
        mse = mean_squared_error(y_test, y_predict)
        r2 = r2_score(y_test, y_predict)

        print("The model has predicted successfuly and accuracy is given below")
        print(f"MAE: {mae}, MSE: {mse}, r2_score : {r2}")

    except Exception as e:
        print(f"There is some problem while evalution : {e}")

default_args = {
    "retries" : 1,
    "retry_delay" : timedelta(minutes = 5)
}
with DAG(
    dag_id = "flight_price_predction",
    default_args = default_args,
    start_date = datetime(2024, 12, 29),
    schedule = "@daily",
    catchup = False 
    ) as dag:

    # task 1
    extract = PythonOperator(
        task_id = "extraction",
        python_callable  = extraction
    )

    #task 2
    transform = PythonOperator(
        task_id = "tranform",
        python_callable =  transformation
    )
     #task 3

    train = PythonOperator(
        task_id = "train",
        python_callable =  training
    )

 # task 4
    evalution = PythonOperator(
        task_id  = "evaluate",
        python_callable =  evaluation )
    
    extract >> transform >> train >> evalution



