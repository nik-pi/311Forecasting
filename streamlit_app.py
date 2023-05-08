import streamlit as st
import pandas as pd
import plotly.graph_objs as go

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
st.title('Forecasting 311 Data')
st.plotly_chart(fig_over_time, config={'displayModeBar': False})
st.plotly_chart(fig_error_over_time, config={'displayModeBar': False})