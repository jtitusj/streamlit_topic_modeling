import streamlit as st
import pandas as pd

file_formats = ["xls", "xlsx", "csv"]


def upload_file(title):
    """
    Uploads xls/xlsx/csv formatted spreadsheet into the app

    Parameters
    ==========
    title: str
        Title to be used in the fill upload section

    Returns
    =======
    streamlit file object
    """
    return st.file_uploader(title, type=file_formats)

@st.cache(allow_output_mutation=True,
         suppress_st_warning=True)
def load_data(data_file):
    show_file = st.empty()

    if not data_file:
        show_file.info("Please upload a file of type: " + ", ".join(file_formats))
        return

    try: 
        return pd.read_csv(data_file)
    except:
        return pd.read_excel(data_file)