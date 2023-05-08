import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import numpy as np



def combine_data() -> pd.DataFrame:
    df_test = pd.read_csv(
        filepath_or_buffer='model/vals.csv',
        index_col='Date'
    ).rename(columns={'Num': 'Actual'})

    df_preds = pd.read_csv(
        filepath_or_buffer='model/preds.csv',
        index_col='Date'
    ).rename(columns={'Num': 'Predicted'})
    df_preds['Predicted'] = df_preds['Predicted'].astype('int')

    df = pd.merge(
        left=df_test,
        right=df_preds,
        left_index=True,
        right_index=True,
        how='inner'
    ).reset_index()

    df['% Difference'] = ((df['Predicted'] - df['Actual']) / df['Actual']).abs()
    print(df['% Difference'])
    return df

def create_over_time_chart(df:pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], name='Actual', ))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], name='Predicted'))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")

    return fig

def mape_heatmap(df: pd.DataFrame) -> go.Figure:
    arr = df['% Difference'].to_numpy()
    num_to_add = 28 - len(arr) % 28
    zeroes = np.zeros(num_to_add)
    arr = np.append(arr, zeroes)
    num_splits = len(arr)/ 28
    arr = np.split(arr, num_splits)
    fig = px.imshow(arr)
    hover_text = []
    for row in arr:
        hover_row = []
        for val in row:
            hover_row.append(f"Value: {val:.2f}") # Customize the hover text format as needed
        hover_text.append(hover_row)

    fig = px.imshow(arr, hovertemplate='%{customdata}', customdata=hover_text)
    fig.update_traces(hovertemplate='%{customdata}') # Update the hover template to display the custom hover text

    return fig
df = combine_data()
fig_over_time = create_over_time_chart(df=df)
fig_mape = mape_heatmap(df = df)

st.title('Forecasting 311 Data')
st.plotly_chart(fig_over_time, config={'displayModeBar': False})
st.plotly_chart(fig_mape)