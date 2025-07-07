import streamlit as st
import io
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

from large_losses import *

###################################

# List of available plots
lista_garantias=['RCVPRIV', 'RCEDIF', 'MPVE', 'CTVE', 'MDCRICTSE', 'ROBCTSE', 'FELCTSE', 'FATOTCTSE', 
                 'FATPCTSE', 'FATLTOTCTSE', 'FATVTOTCTSE', 'DAGCTSE', 'MDCRIMPSE', 'ROBMPSE',  'INCMPSE', 
                 'FELMPSE', 'FATOTMPSE', 'FATPMPSE', 'FATLTOTMPSE', 'FATVTOTMPSE', 'DAGMPSE', 'INCCTSE']

filename = 'C:\\Users\\z105621\\Allianz\\Synapse_Solutions_Hub - 4. MLOps_Hogar\\2025\\1. BBDD\\Large_losses\\Ficheros\\table_ll_analysis.prq'

th_export_file = "thresholds_large_loss_analysis.csv"

###################################

# Dictionary to store x coordinates for each plot
thresholds_dict = {gar: 0 for gar in lista_garantias}

#TODO df to store the cutted nclaims and cost

# load analysis data
df = pd.read_parquet(filename)

# Streamlit UI
st.title("MLApps - Large Loss Analysis")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")

    # Plot selector
    selected_gar = st.selectbox("Select the Cover:", lista_garantias)

    # Input for x coordinate of vertical line
    threshold = st.number_input("Enter the threshold", value=0.0)

    # Button to add x coordinate to the list
    if st.button("Select threshold"):
        thresholds_dict[selected_gar] = threshold
        st.success(f"Selected threshold={threshold} for {selected_gar}")

    # Export button to save x coordinates in CSV file
    if st.button("Export thresholds"):
        # Create a DataFrame from the dictionary
        df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in thresholds_dict.items()]))
        # Convert DataFrame to CSV format
        csv_buffer = StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(label="Download CSV", data=csv_buffer.getvalue(), file_name=th_export_file, mime="text/csv")

th_plot = thresholds_dict.get(selected_gar, 0)

# Show plot on the main page
st.plotly_chart(plot_shape_parameter(df[df["gar"] == selected_gar], threshold=th_plot))
st.plotly_chart(plot_average_excess(df[df["gar"] == selected_gar], threshold=th_plot))
st.plotly_chart(plot_ks_test(df[df["gar"] == selected_gar], threshold=th_plot))
st.plotly_chart(cuts_by_threshold(df[df["gar"] == selected_gar], threshold=th_plot))

