import streamlit as st
import pandas as pd

st.title("Student analytics dashboard")
st.subheader("Load Data")
#upload excel/csv file
uploaded_file = st.file_uploader("Upload student Excel or csv file", type=["csv", "xlsx"]) 
if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.success("file uploaded")
    st.dataframe(df.head())
    # Save the uploaded file for later use
    df.to_csv("student_data.csv", index=False) 
    st.session_state['data_loaded'] = True
    st.session_state['data'] = df
    st.write("Data saved for analysis. Proceed to the next section.")
else:
    st.info("Please upload a file to proceed.")
    st.session_state['data_loaded'] = False

#summary statistics
st.subheader("Summary Statistics")
if st.session_state.get('data_loaded'):
    df = st.session_state['data']
    st.write("Summary statistics of the dataset:")
    st.write(df.describe(include='all'))
else:
    st.info("Upload data to see summary statistics.")
# Display data types
st.subheader("Data Types")
if st.session_state.get('data_loaded'):
    df = st.session_state['data']
    st.write("Data types of each column:")
    st.write(df.dtypes)
else:
    st.info("Upload data to see data types.")
# Display missing values
st.subheader("Missing Values")
if st.session_state.get('data_loaded'):
    df = st.session_state['data']
    st.write("Count of missing values in each column:")
    st.write(df.isnull().sum())
else:
    st.info("Upload data to see missing values.")
