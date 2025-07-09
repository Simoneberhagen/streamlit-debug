# FormatsApp

Define risk factors' formats interactively with a Streamlit based App.

## About the project

This project is a Streamlit application that allows users to interactively define and visualize formats for risk factors. It's designed to help data analysts and modelers to quickly explore different formatting options and their impact on the distribution of variables.

## Getting Started

To get started with this project, you need to have Python and pip installed.

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    ```
2.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Streamlit app:**
    ```bash
    streamlit run format_app.py
    ```

## Features

*   **Interactive format definition:** Users can define formats by specifying the number of bins, distribution, and other parameters.
*   **Real-time visualization:** The app displays a histogram of the formatted variable, which updates in real-time as the user changes the format parameters.
*   **Save and load formats:** Users can save their formats to an Excel file and load them back later.
*   **Categorical variable support:** The app supports both numerical and categorical variables.

## Configuration

The project uses a `config.toml` file to store configuration parameters. This file is located in the root directory of the project.

The following parameters can be configured:

*   `lob`: The line of business.
*   `year`: The year of the data.
*   `path_df`: The path to the Parquet file containing the raw data.
*   `path_formats_table`: The path to the Excel file containing the format tables.
*   `path_figures_dict`: The path to the pickle file containing the figures dictionary.
*   `path_tables_dict`: The path to the pickle file containing the tables dictionary.
*   `path_formats`: The path to the Excel file containing the formats.
*   `path_data_dict`: The path to the Excel file containing the data dictionary.
*   `path_model_dict`: The path to the Excel file containing the model dictionary.
*   `path_formats_dict`: The path to the Excel file containing the formats dictionary.
*   `data_dict_sheet`: The name of the sheet in the data dictionary Excel file.

## Data

The app uses the following data files:

*   `raw_data.prq`: A Parquet file containing the raw data.
*   `formats.xlsx`: An Excel file containing the format tables.
*   `data_dict.xlsx`: An Excel file containing the data dictionary.
*   `model_dict.xlsx`: An Excel file containing the model dictionary.
*   `formats_dict.xlsx`: An Excel file containing the formats dictionary.

These files are located in the `data` and `metadata` directories.