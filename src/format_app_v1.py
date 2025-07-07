import pickle
import pandas as pd
import numpy as np
import streamlit as st
import io

# from src.formats import parse_format_dict_row

univariates_path = r"C:\Users\z105621\Allianz\Synapse_Solutions_Hub - 4. MLOps_Hogar\2025\2. bid&fac\Formatos\univariates_v4.pkl"
formats_path = r"C:\Users\z105621\Allianz\Synapse_Solutions_Hub - 4. MLOps_Hogar\2025\2. bid&fac\Formatos\FORMATOS_NOFAT.xlsx"
data_dict_path = r"C:\Users\z105621\Allianz\Synapse_Solutions_Hub - 4. MLOps_Hogar\2025\2. bid&fac\Diccionarios\data_dict.xlsx"
data_dict_sheet = "2025_all"

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
    with open(univariates_path, 'rb') as file:
        st.session_state.ow_dict = pickle.load(file)

    # Loads format tables from file
    st.session_state.formats_table = pd.read_excel(formats_path)[['START', 'END', 'FMTNAME', 'TYPE', 'LABEL', 'HLO', 'SEXCL', 'EEXCL']]
    # Convert DataFrame to Excel buffer
    st.session_state.excel_data = convert_df_to_excel(st.session_state.formats_table)

    # Factors list
    factors = np.sort(list(st.session_state.ow_dict.keys()))

    # Mapping category-factor
    data_dict = pd.read_excel(data_dict_path, sheet_name=data_dict_sheet)[["CATEGORÍA", "Factores"]]
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

# Streamlit app
st.set_page_config(layout="wide")
st.title("MLApps - Formats Definition")

# Sidebar category selector
selected_cat = st.sidebar.selectbox("Select a Category:", np.sort(list(st.session_state.cover_map.keys())))

# Sidebar toggles for format parameters and table
show_format_params = st.sidebar.toggle("Toggle Format Parameters", value=False)
show_format_table = st.sidebar.toggle("Toggle Format Table", value=False)

# Create a download button
st.sidebar.download_button(
    label="Download Format Table",
    data=st.session_state.excel_data,
    file_name="dataframe.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary"
)

# Dropdown menu to select the factor based on selected category
selected_fac = st.selectbox("Select a factor:", np.sort(st.session_state.cover_map[selected_cat]))

# label = data_dict.loc[data_dict["Factores"]==selected_fac, "LABEL"].item()
# st.write(f"Label: {label}")

# Create the columns
if show_format_params:
    col1, col2 = st.columns([3, 1])
else:
    col1 = st.container()

# Display each plot in its respective column
with col1:
    st.plotly_chart(st.session_state.ow_dict[selected_fac], use_container_width=True)

# Add numerical inputs and dropdowns in col2 if show_format_params is True
if show_format_params:
    with col2:
        # Create two columns for numerical inputs inside col2
        num_col1, num_col2 = st.columns(2)

        # Distribution
        dropdown_options = ["Uniform", "Normal", "Discrete"]
        dropdown_value1 = st.selectbox("Format Distribution", dropdown_options)

        # Number of Bins
        bins_num = st.number_input("Number of Levels", value=32)
        
        with num_col1:
            floor = st.number_input("Min Cap", min_value=0, max_value=100)
            lowest = st.number_input("Min Level", min_value=floor, value=0)
            # Missing Value
            missing_value = st.text_input("Missing Label", value="Missing")
            # NP value
            np_value = st.text_input("NP Label", value="NP")

        with num_col2:
            cap = st.number_input("Max Cap", min_value=0, max_value=100, value=0)
            highest = st.number_input("Min Level", max_value=cap, value=0)
            # Missing Value
            missing_value = st.text_input("Missing Value")
            # NP value
            np_value = st.text_input("NP Value")


if show_format_table:
    format_table_var = st.session_state.formats_table[st.session_state.formats_table["FMTNAME"]=="FMT_"+selected_fac]
    st.data_editor(data=format_table_var)
