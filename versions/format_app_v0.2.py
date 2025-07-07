import pickle
import pandas as pd
import numpy as np
import streamlit as st
import io
import toml

# Load the configuration from the TOML file
config = toml.load('config.toml')

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
if 'ow_dict' not in st.session_state:
    # Load plots from file
    with open(path_figures_dict, 'rb') as file:
        st.session_state.ow_dict = pickle.load(file)

    # Loads format tables from file
    st.session_state.formats_table = pd.read_excel(path_formats)[['START', 'END', 'FMTNAME', 'TYPE', 'LABEL', 'HLO', 'SEXCL', 'EEXCL']]
    # Convert DataFrame to Excel buffer
    st.session_state.excel_data = convert_df_to_excel(st.session_state.formats_table)

    # Factors list
    factors = np.sort(list(st.session_state.ow_dict.keys()))

    # Mapping category-factor
    data_dict = pd.read_excel(path_data_dict, sheet_name=data_dict_sheet)
    data_dict = data_dict[data_dict[f"facbid_DAG"]==1]
    cover_map = {cat: data_dict[data_dict["CATEGORÍA"]==cat]["Factores"].to_list() for cat in data_dict["CATEGORÍA"].unique()}
    
    # Subset the factors per each category according to the available ones
    empty_cats = []
    for cat in cover_map:
        cover_map[cat] = [x for x in cover_map[cat] if x in factors]
        if cover_map[cat] == []:
            empty_cats.append(cat)
    for cat in empty_cats:
        cover_map.pop(cat)
    
    # Store cover_map in session state
    st.session_state.cover_map = cover_map
    st.session_state.data_dict = data_dict


# Streamlit app
st.set_page_config(layout="wide")
st.title("FormatsApp")

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
    st.plotly_chart(st.session_state.ow_dict[selected_fac], use_container_width=True)

# Add numerical inputs and dropdowns in col2 if show_format_params is True
with col2:

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
    format_table_var = st.session_state.formats_table[st.session_state.formats_table["FMTNAME"]=="FMT_"+selected_fac]
    st.data_editor(data=format_table_var)
