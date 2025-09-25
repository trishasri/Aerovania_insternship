import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
st.title("Student Analytics Dashboard")
upload_file = st.file_uploader("Upload student data file", type=["csv", "xlsx"])
if upload_file:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)
    st.success("File uploaded successfully")
    st.dataframe(df.head())
    #sidebar stats
    st.sidebar.subheader("Dataset Statistics")
    st.sidebar.write(f"Average GPA: {df['GPA'].mean():.2f}")
    st.sidebar.write(f"Average Study Time Weekly: {df['StudyTimeWeekly'].mean():.2f} hours")
    st.sidebar.write(f"Average Absences: {df['Absences'].mean():.2f}")
    #create tabs
    tab1, tab2, tab3 = st.tabs(["Data Overview", "Visualizations", "Predictive Modeling"])
    with tab1:
        st.subheader("Data Overview")
        st.write("Summary statistics of the dataset:")
        st.write(df.describe(include='all'))
        st.write("Data types of each column:")
        st.write(df.dtypes)
        st.write("Count of missing values in each column:")
        st.write(df.isnull().sum())
    with tab2:
        st.subheader("Visualizations")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        # Histogram for numeric columns
        st.write("### Histograms for Numeric Columns")
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.histplot(df[col].dropna(), kde=True)
            plt.title(f'Histogram of {col}')
            st.pyplot(plt)
            plt.clf()
        # Bar plots for categorical columns
        st.write("### Bar Plots for Categorical Columns")
        for col in categorical_cols:
            plt.figure(figsize=(8, 4))
            sns.countplot(y=df[col], order=df[col].value_counts().index)
            plt.title(f'Bar Plot of {col}')
            st.pyplot(plt)
            plt.clf()
        # Correlation heatmap
        st.write("### Correlation Heatmap")
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Heatmap')
        st.pyplot(plt)
        plt.clf()
    with tab3:
        df['pass'] = df['GPA'].apply(lambda x: 1 if x >= 2.0 else 0)
        x = df[["GPA","StudyTimeWeekly","GradeClass","Absences"]]
        y = df["pass"] 
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = model.score(x_test, y_test) 
        st.write(f"Model Accuracy: {accuracy:.2f}")
        classification_report = classification_report(y_test, y_pred, target_names=["Fail", "Pass"])
        confusion_mat = confusion_matrix(y_test, y_pred)
        st.write("Classification Report")
        st.text(classification_report)
        st.write("Confusion Matrix")
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', xticklabels=["Fail", "Pass"], yticklabels=["Fail", "Pass"])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(plt)
        plt.clf()
        st.subheader("Predict Pass or Fail")
        GPA_input = st.number_input("Enter GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.1)
        StudyTimeWeekly_input = st.number_input("Enter Study Time Weekly (hours)", min_value=0, max_value=100, step=1)
        Gradeclass_input = st.number_input("Enter Grade Class (1-4)", min_value=1, max_value=4, step=1)
        Absences_input = st.number_input("Enter Number of Absences", min_value=0, max_value=100, step=1)
        if st.button("Predict"):    
            input_data = pd.DataFrame({
                "GPA": [GPA_input],
                "StudyTimeWeekly": [StudyTimeWeekly_input],
                "GradeClass": [Gradeclass_input],
                "Absences": [Absences_input]
            })
            prediction = model.predict(input_data)
            result = "Pass" if prediction[0] == 1 else "Fail"
            st.write(f"The student is predicted to: {result}")