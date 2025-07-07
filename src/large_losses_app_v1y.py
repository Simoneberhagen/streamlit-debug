import streamlit as st
import io
import os
import numpy as np
import plotly.graph_objects as go
import pandas as pd
from io import StringIO

from large_losses import *

###################################

lista_garantias=['RCVPRIV', 'RCEDIF', 'MPVE', 'CTVE', 'MDCRICTSE', 'ROBCTSE', 'FELCTSE', 'FATOTCTSE', 
                 'FATPCTSE', 'FATLTOTCTSE', 'FATVTOTCTSE', 'DAGCTSE', 'MDCRIMPSE', 'ROBMPSE',  'INCMPSE', 
                 'FELMPSE', 'FATOTMPSE', 'FATPMPSE', 'FATLTOTMPSE', 'FATVTOTMPSE', 'DAGMPSE', 'INCCTSE']

folder_path = r"C:\\Users\\z105621\\Allianz\\Synapse_Solutions_Hub - 4. MLOps_Hogar\\2025\\1. BBDD\\Large_losses\\Ficheros\\"
analysis_file = os.path.join(folder_path, "table_ll_analysis.prq")
thresholds_file = os.path.join(folder_path, "thresholds_large_loss_analysis.csv")

###################################

# Function to convert DataFrame to Excel
def convert_df_to_excel(df):
    # Create a BytesIO buffer to hold the Excel file
    buffer = io.BytesIO()
    # Write the DataFrame to the buffer using ExcelWriter
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Sheet1')
    # Seek to the beginning of the stream
    buffer.seek(0)
    return buffer

cols = ["thresholds", "Total Cost", "Cost Above", "Perc Cost Above", "Total Claims", "Claims Above", "Perc Claims Above"]

if 'df_cuts' not in st.session_state:
    if os.path.exists(thresholds_file):
        st.session_state.df_cuts = pd.read_csv(thresholds_file)
    else:
        st.session_state.df_cuts = pd.DataFrame(index=lista_garantias, columns=cols)
        st.session_state.df_cuts["thresholds"] = 0

# load analysis data
df = pd.read_parquet(analysis_file)

# Streamlit UI
st.title("MLApps - Large Loss Analysis")

# Sidebar for inputs
with st.sidebar:
    st.header("Settings")

    # Cover selector
    selected_gar = st.selectbox("Select the Cover:", lista_garantias)
    df_gar = df[df["gar"] == selected_gar]

    # Threshold selector
    threshold = st.number_input("Enter the threshold", value=st.session_state.df_cuts.loc[selected_gar, "thresholds"])
    st.session_state.df_cuts.loc[selected_gar, "thresholds"] = threshold

    # Button to save threshold
    if st.button("Export threshold"):
        st_session_state.df_cuts.to_excel(thresholds_file)


selected_th = st.session_state.df_cuts.loc[selected_gar, "thresholds"]

# selected_th=0

# Show plot on the main page
st.plotly_chart(plot_shape_parameter(df_gar, threshold=selected_th))
st.plotly_chart(plot_average_excess(df_gar, threshold=selected_th))
st.plotly_chart(plot_ks_test(df_gar, threshold=selected_th))
st.plotly_chart(cuts_by_threshold(df_gar, threshold=selected_th))

