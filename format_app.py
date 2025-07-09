import io
import toml
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import gpc_utils.sas as su
import pyarrow.parquet as pq
from src.univariate import univariate_plotly
from src.formats import define_format, parse_format_dict_row


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
    st.session_state.formats_table = pd.read_excel(path_formats, dtype={'START': str, 'END': str})[['START', 'END', 'FMTNAME', 'TYPE', 'LABEL', 'HLO', 'SEXCL', 'EEXCL']]
    # Convert DataFrame to Excel buffer
    st.session_state.excel_data = convert_df_to_excel(st.session_state.formats_table)

    # Pyarrow parquet file with raw data
    st.session_state.pq_file = pq.ParquetFile(path_df)

    # Mapping category-factor
    data_dict = pd.read_excel(path_data_dict, sheet_name=data_dict_sheet)
    data_dict = data_dict[data_dict[f"facbid_DAG"]==1]
    cover_map = {cat: data_dict[data_dict["CATEGORÍA"]==cat]["Factores"].to_list() for cat in data_dict["CATEGORÍA"].unique()}
    
    # Store cover_map in session state
    st.session_state.cover_map = cover_map
    st.session_state.data_dict = data_dict
    st.session_state.formats_dict = pd.read_excel(path_formats_dict)
    st.session_state.edited_format_table = None


# Streamlit app
st.set_page_config(layout="wide")
st.title(f"FormatsApp - {lob} {year}")

# Model selector (for weight and response shown in the plot)
model_dict = pd.read_excel(path_model_dict).set_index("model")
selected_model = st.sidebar.selectbox("Select a Model:", model_dict.index)
weight = model_dict.loc[selected_model, "exp"]
resp = model_dict.loc[selected_model, "response"]

# get data in memory from the spark connection
if "pq_file" in st.session_state and st.session_state.pq_file is not None:
    df_var_full = st.session_state.pq_file.read(use_pandas_metadata=True).to_pandas()
else:
    st.error("Data file not loaded. Please restart the app or check the file path in config.toml.")
    st.stop()

# Sidebar category selector
selected_cat = st.sidebar.selectbox("Select a Category:", np.sort(list(st.session_state.cover_map.keys())))
factors_by_cat = st.session_state.cover_map.get(selected_cat)
data_dict = st.session_state.data_dict[st.session_state.data_dict.Factores.isin(factors_by_cat)]
labels_to_factors = {row["LABEL"]: row["Factores"] for _, row in data_dict.iterrows()}

# Dropdown menu to select the factor based on selected category
selected_fac = labels_to_factors[st.sidebar.selectbox("Select a factor:", labels_to_factors.keys())]

# format the in-memory dataframe and generate a plot
fmt_table = st.session_state.formats_table[st.session_state.formats_table.FMTNAME=="FMT_"+selected_fac]
df_var = df_var_full[[selected_fac, resp, weight]].copy()
df_var[selected_fac+"_formatted"] = su.apply_format(vec=df_var[selected_fac], fmt_table=fmt_table)

# Calculate univariate table to get levels and weights
univariate_df, _ = univariate_plotly(df_var, x=selected_fac+"_formatted", y=resp, w=weight, output=True, show_fig=False, retfig=True)
univariate_table = univariate_df[selected_fac+"_formatted"][0]

# Determine default base level
non_base_labels = ["Missing", "Other", "NP"]
eligible_levels = univariate_table[~univariate_table['label'].isin(non_base_labels)]
if not eligible_levels.empty:
    default_base_level = eligible_levels.loc[eligible_levels[weight].idxmax()]['label']
else:
    default_base_level = None

# Sidebar controls for base level
lock_base_level = st.sidebar.checkbox("Lock Base Level", value=True)

if lock_base_level:
    selected_base_level = default_base_level
    st.sidebar.selectbox("Base Level", [selected_base_level] if selected_base_level else [], disabled=True)
else:
    level_options = list(univariate_table['label'])
    filtered_options = [level for level in level_options if level not in non_base_labels]
    
    index = 0
    if default_base_level in filtered_options:
        index = filtered_options.index(default_base_level)

    selected_base_level = st.sidebar.selectbox("Base Level", filtered_options, index=index, disabled=False)


# Sidebar toggles for format parameters and table
edit_format_table = st.sidebar.toggle("Edit Format Table", value=False)
view_mode = st.sidebar.radio("View Mode", ["Graph", "Table"], index=0)

# Main layout with two columns
col1, col2 = st.columns([2, 1])

with col1:
    # Display each plot in its respective column
    required_cols = [selected_fac + "_formatted", resp, weight]
    missing_cols = [col for col in required_cols if col not in df_var.columns]
    table, fig = univariate_plotly(df_var, x=selected_fac+"_formatted", y=resp, fig_title=data_dict[data_dict.Factores==selected_fac]["LABEL"].item(),
                                       w=weight, w_name=weight, base_level=selected_base_level, fig_w=1100, fig_h=700, retfig=True, show_fig=False, output=True)

    if view_mode == "Graph":
        st.plotly_chart(fig, use_container_width=True)
    else:
        univariate_table = table[selected_fac+"_formatted"][0].copy()
        univariate_table = univariate_table.reset_index()
        
        new_names = {
            "label": "Label",
            weight: f"Weight % ({selected_model})",
            f"{weight}_Sum": f"Total Weight ({selected_model})",
            resp: f"Avg. {resp} ({selected_model})"
        }
        univariate_table.rename(columns=new_names, inplace=True)
        
        display_columns = ["Label", f"Total Weight ({selected_model})", f"Weight % ({selected_model})", f"Avg. {resp} ({selected_model})"]
        
        st.dataframe(univariate_table[display_columns], use_container_width=True)

with col2:
    st.subheader("Format Parameters")

    dropdown_value1 = st.selectbox("Format Distribution", dropdown_options, 
                                   index=dropdown_options.index(dist_val),
                                   disabled=edit_format_table or is_categorical)

    num_bins_val = factor_params.get("num_bins", 32)
    if pd.isna(num_bins_val):
        num_bins_val = 32
    bins_num = st.number_input("Number of Levels", value=int(num_bins_val), disabled=edit_format_table or is_categorical)
    
    num_decimals_val = factor_params.get("num_decimals", 0)
    if pd.isna(num_decimals_val):
        num_decimals_val = 0
    num_decimals = st.number_input("Number of Decimals", value=int(num_decimals_val), disabled=edit_format_table or is_categorical)

    floor = st.number_input("Min Value", value=factor_params.get("floor", np.nan), disabled=edit_format_table or is_categorical)
    lowest = st.number_input("Min Level", value=factor_params.get("lowest", np.nan), min_value=floor if not pd.isna(floor) else None, disabled=edit_format_table or is_categorical)

    cap = st.number_input("Max Value", value=factor_params.get("cap", np.nan), disabled=edit_format_table or is_categorical)
    highest = st.number_input("Max Level", value=factor_params.get("highest", np.nan), max_value=cap if not pd.isna(cap) else None, disabled=edit_format_table or is_categorical)

    missing_values_str = st.text_input("Missing Values (comma-separated)", value="", disabled=edit_format_table)
    np_values_str = st.text_input("NP Values (comma-separated)", value="", disabled=edit_format_table)

    # Button to save the changes to the format and plot the updated univariate
    if st.button("Update Format"):
        if edit_format_table:
            if st.session_state.edited_format_table is not None:
                st.session_state.formats_table = st.session_state.formats_table[st.session_state.formats_table.FMTNAME!="FMT_"+selected_fac]
                st.session_state.formats_table = pd.concat([st.session_state.formats_table, st.session_state.edited_format_table])
                st.session_state.edited_format_table = None
                st.rerun()
        elif not is_categorical:
            # Parse missing values
            if missing_values_str:
                missing_values = [float(x.strip()) for x in missing_values_str.split(",")]
            else:
                missing_values = []

            # Parse NP values
            if np_values_str:
                np_values = [float(x.strip()) for x in np_values_str.split(",")]
            else:
                np_values = []

            params = {
                "distribution": dropdown_value1.lower(),
                "num_bins": bins_num,
                "num_decimals": num_decimals,
                "floor": floor,
                "lowest": lowest,
                "cap": cap,
                "highest": highest,
                "missing_values": missing_values,
                "np_values": np_values
            }
            
            df_var = st.session_state.pq_file.read(columns=[selected_fac, resp, weight], use_pandas_metadata=True).to_pandas()
            
            new_format = define_format(df=df_var, var=selected_fac, weight=weight, **params)
            
            st.session_state.formats_table = st.session_state.formats_table[st.session_state.formats_table.FMTNAME!="FMT_"+selected_fac]
            st.session_state.formats_table = pd.concat([st.session_state.formats_table, new_format])
            st.rerun()

    # Button to save the changes to the format and plot the updated univariate
    if st.button("Save Parameters"):
        
        # Get the index of the selected factor
        idx = st.session_state.formats_dict[st.session_state.formats_dict.factor == selected_fac].index
        
        params_to_save = {
            "factor": selected_fac,
            "distribution": dropdown_value1.lower(),
            "num_bins": bins_num,
            "num_decimals": num_decimals,
            "floor": floor,
            "lowest": lowest,
            "cap": cap,
            "highest": highest
        }

        if idx.empty:
            # Add a new row for the new factor
            new_row_df = pd.DataFrame([params_to_save])
            st.session_state.formats_dict = pd.concat([st.session_state.formats_dict, new_row_df], ignore_index=True)
        else:
            # Update the parameters in the DataFrame for the existing factor
            for key, value in params_to_save.items():
                if key != 'factor': # 'factor' is for lookup and should not be overwritten
                    st.session_state.formats_dict.loc[idx, key] = value
        
        # Save the updated DataFrame to Excel
        st.session_state.formats_dict.to_excel(path_formats_dict, index=False)
        
        if edit_format_table and st.session_state.edited_format_table is not None:
            st.session_state.formats_table = st.session_state.formats_table[st.session_state.formats_table.FMTNAME!="FMT_"+selected_fac]
            st.session_state.formats_table = pd.concat([st.session_state.formats_table, st.session_state.edited_format_table])
            st.session_state.edited_format_table = None

        st.session_state.formats_table.to_excel(path_formats, index=False)
        st.sidebar.success("Parameters and formats saved successfully!")


# Create a download button
st.sidebar.download_button(
    label="Download Format Table",
    data=convert_df_to_excel(st.session_state.formats_table),
    file_name="formats_table.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    type="primary"
)

if edit_format_table:
    st.subheader("Format Table")
    format_table_var = st.session_state.formats_table[st.session_state.formats_table["FMTNAME"]=="FMT_"+selected_fac]
    st.session_state.edited_format_table = st.data_editor(data=format_table_var)
