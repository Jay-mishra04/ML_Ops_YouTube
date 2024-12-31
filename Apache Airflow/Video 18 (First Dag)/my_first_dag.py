from airflow import DAG
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

# task 1
def greet():
    print("hello how are you")

# task 2
def my_self():
    print("my name is apache airflow")

default_args = {
    "owner" : "Mritunjay",
    "retries" : 1,
    "retry_delay" : timedelta(minutes = 5),
    "start_date" : datetime(2024, 12, 27)

}

with DAG (
    dag_id = "my_self_dag", 
    default_args = default_args, 
    schedule_interval = "@daily",
) as dag:
    
    # first task callout

    greet_task = PythonOperator(
        task_id = "task1",
        python_callable = greet
    )

# second task callout
    self_task = PythonOperator(
        task_id = "task2",
        python_callable = my_self
    )

    greet_task >> self_task
