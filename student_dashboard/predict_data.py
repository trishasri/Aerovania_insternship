import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

st.title("Student Analytics Dashboard")
uploasd_file = st.file_uploader("Upload student data file", type=["csv", "xlsx"])
if uploasd_file:
    if uploasd_file.name.endswith(".csv"):
        df = pd.read_csv(uploasd_file)
    else:
        df = pd.read_excel(uploasd_file)
    st.success("File uploaded successfully")
    st.dataframe(df.head())
    
    # Save the uploaded file for later use
    df.to_csv("student_data.csv", index=False)
    st.session_state['data_loaded'] = True
    st.session_state['data'] = df
    st.write("Data saved for analysis. Proceed to the next section.")

    #create pass or fail column
    df['pass'] = df['GPA'].apply(lambda x: 1 if x >= 2.0 else 0)

    #train test split
    x = df[["GPA","StudyTimeWeekly","GradeClass","Absences"]]
    y = df["pass"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    #train model
    model = RandomForestClassifier()
    model.fit(x_train, y_train)

    #test accuracy
    y_pred = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    # prediction
    st.subheader(" predict pass or fail")
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