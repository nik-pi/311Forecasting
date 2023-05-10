import requests
import pandas as pd
from dotenv import load_dotenv
import os

# load_dotenv()
APP_TOKEN = os.getenv('APP_TOKEN')

def _get_max_date() -> str:
    df = pd.read_csv('model/vals.csv')
    date_str = df.iloc[-1, 0]
    date_str += 'T00:00:00.000'
    return date_str

def _construct_query() -> str:
    url = ' https://data.cityofnewyork.us/resource/erm2-nwe9.json'
    query_app_token = f'?$$app_token={APP_TOKEN}'
    query_select = '&$SELECT=date_trunc_ymd(created_date) as Date, count(*) as Num&$GROUP=date_trunc_ymd(created_date)'
    query_where = f'&$WHERE=created_date >= "{_get_max_date()}"'
    query_order = '&$ORDER=date_trunc_ymd(created_date)'
    query = f'{url}{query_app_token}{query_select}{query_where}{query_order}'
    
    return query

def _filter_bad_data(
        data: list[dict]
        ) -> list[dict]:
    if int(data[-1]['Num']) < 3_000:
        print(data)
        data.pop()

    return data

def update_data() -> None:
    query = _construct_query()
    resp = requests.get(query).content
    with open('log.txt', 'w') as file:
        file.write(resp)
    # if resp:
    #     resp = _filter_bad_data(resp)
    # df = pd.DataFrame(resp)
    # df['Date'] = pd.to_datetime(df['Date'])
    # df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    # old = pd.read_csv('model/vals.csv')
    # df = pd.concat([old, df], ignore_index=True)
    # df = df.drop_duplicates(subset=['Date'], keep='first')
    # df.to_csv('model/vals.csv', index=False)


if __name__ == "__main__":
    update_data()