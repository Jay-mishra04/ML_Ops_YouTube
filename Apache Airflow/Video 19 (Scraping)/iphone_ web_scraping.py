from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests

def scraping():
    from bs4 import BeautifulSoup
    import requests
    import pandas as pd

    url = "https://www.flipkart.com/search?q=iphone&otracker=search&otracker1=search&marketplace=FLIPKART&as-show=on&as=off"

    response = requests.get(url)
    soup = BeautifulSoup(response.text, "lxml")

    brand =[]
    Price = []

    name = soup.find_all("div", class_ = "KzDlHZ")
    for i in name:
        brand_name = i.text.strip()
        brand.append(brand_name)

    price = soup.find_all("div", class_ = "Nx9bqj _4b5DiR")
    for j in price:
        raw_price = j.text.strip()
        # Clean the price: remove '₹', ',' and convert to integer
        clean_price = int(raw_price.replace('₹', '').replace(',', ''))
        Price.append(clean_price)

    data = {'Brand' : brand, f"Price at {datetime.now().strftime('%H:%M')}" : Price}

    df = pd.DataFrame(data)

    path = '/opt/airflow/dags/iphone_price.csv'
    try: 
        old_file = pd.read_csv(path)
        merged_file = pd.merge(old_file, df, on= "Brand", how = "left")
    except FileNotFoundError:
        merged_file = df

    merged_file.to_csv(path, index=False)

def minimum_price():
    # Load the existing file
    file_path = '/opt/airflow/dags/iphone_price.csv'
    file = pd.read_csv(file_path)
    
    # Extract only price columns (dynamically named like "Price at HH:MM")
    price_columns = [col for col in file.columns if col.startswith("Price at")]
    
    # Create a new DataFrame with "Brand" and the minimum price
    new_df = file[["Brand"]].copy()  # Retain only the "Brand" column
    new_df["Minimum Price"] = file[price_columns].min(axis=1)  # Calculate the minimum price per row

    # Save the result back to the CSV or a new file
    new_file_path = '/opt/airflow/dags/iphone_price_min.csv'
    new_df.to_csv(new_file_path, index=False)

with DAG(
     dag_id="iphone_scraping",
     start_date=datetime(2024, 12, 28),
     schedule_interval = "* * * * *",
     catchup = False
 ) as dag:
    
    # task call
    scraping_task = PythonOperator(
        task_id = "scraping_iphone",
        python_callable = scraping
    )

    minimum_price_task = PythonOperator(
        task_id = "minimum_price",
        python_callable = minimum_price
    )

    scraping_task >> minimum_price_task