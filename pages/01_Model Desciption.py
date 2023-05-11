import streamlit as st
import streamlit.components.v1 as components

def mermaid(code: str) -> None:
    components.html(
        f"""
        <pre class="mermaid">
            {code}
        </pre>

        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({{ startOnLoad: true }});
        </script>
        """
    )

st.set_page_config(
    page_title="Model Description",
    page_icon="👨🏻‍💻",
)

st.title('Model Description')
st.header('Data Updating')
st.markdown("""
The 311 Service Request Forecasting Model is a workflow that utilizes Github actions and Python scripts to predict future 311 service requests in New York City. The workflow consists of the following steps:
1. Data is pulled from the 311 Open Data Dataset on the New York City website via the API. This is done using a Python script triggered by a Github action.
2. The new data is saved to Github into a CSV file located in the `data` directory. This ensures that we have a record of all the data we've pulled and can use it for future analysis.
3. Every 7 days, Github actions trigger a forecasting script which runs a SKlearn random forest regressor to recursively forecast data. This script uses the data in the `data` directory to generate forecasts.
4. The forecasts and accuracy are displayed on a streamlit dashboard. This dashboard is easy to use and provides a great visual representation of the data.

This workflow is designed to be easy to use and accurate. It provides a great tool for anyone interested in predicting 311 service requests in New York City. The workflow can be customized to fit specific needs and can be modified to include additional data sources or forecasting methods.
""")
flow_chart1 = mermaid(
    """
    flowchart LR
        id1[(311 Open Data)]-->id2[Github Action]-->id3[(CSV File)]-->id4[Streamlit Dashboard]
    """
)
st.markdown("""
## Time Series Feature Engineering with CallData Class
The CallData class is designed to engineer features for a time series forecasting model. It takes in a Pandas DataFrame, a list of lags, and a list of windows as arguments and generates a variety of features that can be used to train a time series forecasting model.

### Date Features
Next, **date features** are added to the DataFrame. These include the day of the week, month, week of the year, day of the year, and quarter. The DataFrame is also checked for holidays using the NYSE holiday calendar. The **WorkingDayWithHoliday** feature is added to the DataFrame, which is a binary variable that indicates whether the day is a working day (Monday-Friday) and is not a holiday.

### Lagged Features
The class then calculates **lagged features** for the target variable. This involves shifting the target variable by a specified number of time steps and adding it as a new feature in the DataFrame.

### Rolling Statistics Features
The class also generates **rolling statistics features** for the target variable. This involves calculating rolling means, standard deviations, and medians for a specified number of time steps. These rolling statistics can be split by the **WorkingDayWithHoliday** feature.

### One-Hot Encoded Features
The class also performs **one-hot encoding** on the Weekday and Quarter features. This involves creating binary variables for each possible value of the feature.

### Cyclical Features
Finally, the class calculates **cyclical features** for the Month, WeekOfYear, and DayOfYear features. This involves transforming these features into sine and cosine functions, which can be used to capture cyclical patterns in the data.
""")