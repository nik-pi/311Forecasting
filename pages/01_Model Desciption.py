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

st.title('Model Description')
st.header('Data Updating')
st.markdown('Data are updated daily using a Github action. The action triggers a Python script which aggregates data from the Open Data Portal\'s API. The data are then pushed into the values tables in the repository.')
flow_chart1 = mermaid(
    """
    flowchart LR
        id1[(311 Open Data)]-->id2[Github Action]-->id3[(CSV File)]-->id4[Streamlit Dashboard]
    """
)
