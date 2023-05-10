import streamlit as st
import pandas as pd
import plotly.graph_objs as go

def combine_data() -> pd.DataFrame:
    df_test = pd.read_csv(
        filepath_or_buffer='model/vals.csv',
        parse_dates=['Date']
    ).rename(columns={'Num': 'Actual'})
    df_test.index = pd.to_datetime(df_test.index)

    df_preds = pd.read_csv(
        filepath_or_buffer='model/preds.csv',
        index_col='Date',
    ).rename(columns={'Num': 'Predicted'})
    df_preds['Predicted'] = df_preds['Predicted'].astype('int')
    df_preds.index = pd.to_datetime(df_preds.index)

    df = pd.merge(
        left=df_test,
        right=df_preds,
        left_index=True,
        right_index=True,
        how='inner'
    ).reset_index()

    df['% Difference'] = ((df['Predicted'] - df['Actual']) / df['Actual']).abs()
    return df

def create_over_time_chart(df:pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Actual'], name='Actual'))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Predicted'], name='Predicted'))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        x=0.35))
    fig.update_layout(title="Actual vs. Predicted Values")

    
    return fig

def create_error_over_time_chart(df:pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    df['Error'] = (df['Actual'] - df['Predicted']) / df['Actual']
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Error'], name='Error'))
    fig.update_traces(mode="markers+lines", hovertemplate=None)
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(rangemode="tozero")
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        x=0.35))
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(title="Error over Time")

    
    return fig


df = combine_data()
fig_over_time = create_over_time_chart(df=df)
fig_error_over_time = create_error_over_time_chart(df=df)

st.set_page_config(
    page_title="311 Forecasting",
    page_icon="📈",
)

st.title('Forecasting 311 Data')
st.markdown("""
These graphs show forecasted values of 311 Volumes in New York City. Data are provided daily by the city via their [Open Data Program](https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9).

The data are refreshed daily and new forecasted are created monthly for the duration of the month. 

You can check out the entire source code on my [Github page](https://github.com/nik-pi/311Forecasting). 

""")

st.plotly_chart(fig_over_time, config={'displayModeBar': False})
st.plotly_chart(fig_error_over_time, config={'displayModeBar': False})
st.dataframe(df)