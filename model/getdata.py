import requests
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
APP_TOKEN = os.getenv('APP_TOKEN')

def get_max_date() -> str:
    df = pd.read_csv('model/vals.csv')
    date_str = df.iloc[-1, 0]
    date_str += 'T00:00:00.000'
    return date_str

def construct_query() -> str:
    url = ' https://data.cityofnewyork.us/resource/erm2-nwe9.json'
    query_app_token = f'?$$app_token={APP_TOKEN}'
    query_select = '&$SELECT=date_trunc_ymd(created_date) as Date, count(*) as Num&$GROUP=date_trunc_ymd(created_date)'
    query_where = f'&$WHERE=created_date >= "{get_max_date()}"'
    query_order = '&$ORDER=date_trunc_ymd(created_date)'
    query = f'{url}{query_app_token}{query_select}{query_where}{query_order}'
    
    return query

def filter_bad_data(
        data: list[dict]
        ) -> list[dict]:
    
    if int(data[-1]['Num']) < 1_000:
        data.pop()

    return data


query = construct_query()
resp = requests.get(query).json()
resp = filter_bad_data(resp)
print(resp)