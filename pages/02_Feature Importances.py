import streamlit as st
import pandas as pd
import plotly.graph_objs as go

df = pd.read_csv('pages/importances.csv')
df = df.sort_values(by='Importance', ascending=True)

fig = go.Figure()
fig.add_trace(
    go.Bar(
        y=df['Feature'], 
        x=df['Importance'], 
        name='Feature Importance', 
        orientation='h'
        )
    )

fig.update_layout(
    title="Feature Importances",
    xaxis=dict(tickformat='.1%'),
    height=600,
    )

st.set_page_config(
    page_title="Feature Importances",
    page_icon="🤖",
)

text = """
In this model, I used a Random Forest Regressor and was able to collect the feature importances of the model. The results provided valuable insights into which features contributed the most to the model's predictions. Here's a summary of the feature importances obtained from the Random Forest Regressor:

The most important featuress turned out to be:
1. *Rolling_WorkingDayWithHoliday_5_mean*: 12.6%
2. *Rolling_WorkingDayWithHoliday_10_median*: 12.6%
3. *Rolling_WorkingDayWithHoliday_15_median*: 11.2%. 

**This indicates the importance of regressive features.**
"""

supplemental = """
Feature importances are a crucial aspect of understanding and interpreting machine learning models, particularly when working with complex algorithms like Random Forest or Gradient Boosting Machines. They provide a quantitative measure of how much each feature contributes to the model's predictions. In this article, we'll discuss how to interpret feature importances, why they matter, and their limitations.

## How to Interpret Feature Importances

Feature importances are usually represented as a score or percentage, indicating the relative importance of each feature in the model. A higher score means that the feature has a more significant impact on the model's predictions. Feature importances are relative, so comparing the scores across features is essential to identify which ones have a more substantial impact on the model's performance.

## Why Feature Importances Matter

Understanding feature importances is crucial for several reasons:

1. _Model Interpretability_: Knowing which features are the most important helps explain the model's predictions and makes it more interpretable for stakeholders.
2. _Feature Selection_: Identifying the most important features can guide the feature selection process, allowing you to remove irrelevant or redundant features, thus reducing the model's complexity and improving its performance.
3. _Feature Engineering_: Understanding the influential features can help you focus on creating new, relevant features that may improve the model's performance further.
4. _Model Debugging_: If an unexpected feature has a high importance score, it may indicate an issue with the data or the model, such as data leakage or overfitting.

## Limitations of Feature Importances

Despite their usefulness, feature importances have some limitations:

1. _Correlated Features_: If two or more features are highly correlated, the importance scores may be distributed among them, making it difficult to determine the true importance of each feature.
2. _Model-Specific_: Feature importances are specific to the model used and may vary across different models. Therefore, it's essential to consider the model's performance and assumptions when interpreting feature importances.
3. _No Causality_: Feature importances only indicate the strength of the relationship between the feature and the target variable, not causality. A high importance score doesn't necessarily imply that the feature causes the target variable to change.

In conclusion, feature importances are a valuable tool for understanding and interpreting machine learning models. They help identify the most influential features, guide feature selection and engineering efforts, and improve model interpretability. However, it's essential to be aware of their limitations and consider the specific model and data context when interpreting their results.
"""

st.title('Feature Importances')
st.markdown(text)
st.plotly_chart(fig)
st.markdown(supplemental)