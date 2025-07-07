import io
import toml
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import gpc_utils.sas as su
import pyarrow.parquet as pq
from src.univariate import univariate_plotly


# Load the configuration from the TOML file
config = toml.load('config.toml')
lob = config['project']['lob']
year = config['project']['year']

# Access the data paths
path_df = config['data']['path_df']
path_formats_table = config['data']['path_formats_table']
path_figures_dict = config['data']['path_figures_dict']
path_tables_dict = config['data']['path_tables_dict']
path_formats = config['data']['path_formats']

# Access the dictionaries paths
path_data_dict = config['dictionaries']['path_data_dict']
path_model_dict = config['dictionaries']['path_model_dict']
path_formats_dict = config['dictionaries']['path_formats_dict']
data_dict_sheet = config['dictionaries']['data_dict_sheet']

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

# Load data only if not already loaded
if 'formats_table' not in st.session_state:

    # Loads format tables from file
    st.session_state.formats_table = pd.read_excel(path_formats)[['START', 'END', 'FMTNAME', 'TYPE', 'LABEL', 'HLO', 'SEXCL', 'EEXCL']]
    # Convert DataFrame to Excel buffer
    st.session_state.excel_data = convert_df_to_excel(st.session_state.formats_table)

    # Pyarrow parquet file with raw data
    # st.session_state.pq_file = pq.ParquetFile(path_df)

    # Mapping category-factor
    data_dict = pd.read_excel(path_data_dict, sheet_name=data_dict_sheet)
    data_dict = data_dict[data_dict[f"facbid_DAG"]==1]
    cover_map = {cat: data_dict[data_dict["CATEGORÍA"]==cat]["Factores"].to_list() for cat in data_dict["CATEGORÍA"].unique()}
    
    # Store cover_map in session state
    st.session_state.cover_map = cover_map
    st.session_state.data_dict = data_dict


# Streamlit app
st.set_page_config(layout="wide")
st.title(f"FormatsApp - {lob} {year}")

# Model selector (for weight and response shown in the plot)
model_dict = pd.read_excel(path_model_dict).set_index("model")
selected_model = st.sidebar.selectbox("Select a Model:", model_dict.index)
weight = model_dict.loc[selected_model, "exp"]
resp = model_dict.loc[selected_model, "response"]
##TODO remove
path_df = f"C:\\Users\\z105621\\Allianz\\Synapse_Solutions_Hub - 4. MLOps_Hogar\\2025\\2. bid&fac\\Data\\{model_dict.loc[selected_model, 'cover']}.parquet"
st.session_state.pq_file = pq.ParquetFile(path_df)

# Sidebar category selector
selected_cat = st.sidebar.selectbox("Select a Category:", np.sort(list(st.session_state.cover_map.keys())))
factors_by_cat = st.session_state.cover_map.get(selected_cat)
data_dict = st.session_state.data_dict[st.session_state.data_dict.Factores.isin(factors_by_cat)]
labels_to_factors = {row["LABEL"]: row["Factores"] for _, row in data_dict.iterrows()}

# Dropdown menu to select the factor based on selected category
selected_fac = labels_to_factors[st.sidebar.selectbox("Select a factor:", labels_to_factors.keys())]

# Sidebar toggles for format parameters and table
edit_format_table = st.sidebar.toggle("Edit Format Table", value=False)

# Button to save the changes to the format and plot the updated univariate
if st.sidebar.button("Update Format"):
    if edit_format_table:
        pass
    else:
        pass

# Create a download button
st.sidebar.download_button(
    label="Download Format Table",
    data=st.session_state.excel_data,
    file_name="dataframe.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary"
)

# Create the columns for the main part of the app
col1, col2 = st.columns([3, 1])

# Display each plot in its respective column
with col1:
    # get data in memory from the spark connection
    df_var = st.session_state.pq_file.read(columns=[selected_fac, resp, weight], use_pandas_metadata=True).to_pandas()
        
    # format the in-memory dataframe and generate a plot
    fmt_table = st.session_state.formats_table[st.session_state.formats_table.FMTNAME=="FMT_"+selected_fac]
    df_var[selected_fac+"_formatted"] = su.apply_format(vec=df_var[selected_fac], fmt_table=fmt_table)
    table, fig = univariate_plotly(df_var, x=selected_fac+"_formatted", y=resp, fig_title=data_dict[data_dict.Factores==selected_fac]["LABEL"].item(),
                                       w=weight,fig_w=1100, fig_h=700, retfig=True, show_fig=False, output=True)

    st.plotly_chart(fig, use_container_width=True)

# Add numerical inputs and dropdowns in col2 if show_format_params is True
with col2:

    st.subheader("Format Parameters")
    st.empty()  # Creates a blank space
    
    # Distribution
    dropdown_options = ["Uniform", "Normal", "Discrete"]
    dropdown_value1 = st.selectbox("Format Distribution", dropdown_options, disabled=edit_format_table)

    # Number of Bins
    bins_num = st.number_input("Number of Levels", value=32, disabled=edit_format_table)

    # Create two columns for numerical inputs inside col2
    num_col1, num_col2 = st.columns(2)

    with num_col1:
        floor = st.number_input("Min Value", disabled=edit_format_table)
        lowest = st.number_input("Min Level", min_value=floor, disabled=edit_format_table)
        # Missing Value
        missing_value = st.text_input("Missing Label", value="Missing", disabled=edit_format_table)
        # NP value
        np_value = st.text_input("NP Label", value="NP", disabled=edit_format_table)

    with num_col2:
        cap = st.number_input("Max Value", disabled=edit_format_table)
        highest = st.number_input("Max Level", max_value=cap, disabled=edit_format_table)
        # Missing Value
        missing_value = st.text_input("Missing Value", disabled=edit_format_table)
        # NP value
        np_value = st.text_input("NP Value", disabled=edit_format_table)

if edit_format_table:
    st.subheader("Format Table")
    format_table_var = st.session_state.formats_table[st.session_state.formats_table["FMTNAME"]=="FMT_"+selected_fac]
    st.data_editor(data=format_table_var)
