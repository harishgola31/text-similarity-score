import streamlit as st
import pandas as pd
import numpy as np
from model import similarity
from helper import FileDownloader

# Title
st.title("Text Similarity Analysis !!!")
menu = ['Input Texts', 'Upload Dataset']
choice = st.sidebar.selectbox("Menu", menu)

if choice == 'Input Texts':
    # Text Input
    text1 = st.text_area("##### Text1: #####")
    text2 = st.text_area("##### Text2: #####")

    data = pd.DataFrame()
    data['text1'] = [text1]
    data['text2'] = [text2]

    # display the name when the submit button is clicked
    # .title() is used to get the input text string
    if(st.button('###### Analyze the Similarity ######')):
        result_df = similarity(data)
        result = result_df['cos_similarity']
        f"### Similarity Score: {result[0]}"
        # st.markdown('Similarity Score: ', result[0])
        download = FileDownloader(result_df.to_csv(), file_ext='csv').download()

elif choice == "Upload Dataset":
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    
    if data_file is not None:

        file_details = {"filename": data_file.name, "filetype": data_file.type,
                        "filesize": data_file.size}
        
        st.write(file_details)
        '##### Please wait, It may take some time...'
        data = pd.read_csv(data_file)
        result_df = similarity(data)
        st.dataframe(result_df.head())
        download = FileDownloader(result_df.to_csv(), file_ext='csv').download()

