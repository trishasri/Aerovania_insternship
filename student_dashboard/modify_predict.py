import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Title
st.title("Student Analytics Dashboard")

# File Upload
upload_file = st.file_uploader("Upload student data file", type=["csv", "xlsx"])

if upload_file:
    if upload_file.name.endswith(".csv"):
        df = pd.read_csv(upload_file)
    else:
        df = pd.read_excel(upload_file)

    st.success("File uploaded successfully")
    st.dataframe(df.head())
    # Sidebar Key Stats
    st.sidebar.header("Quick Stats")
    st.sidebar.write(f"Average GPA:{df['GPA'].mean():.2f}")
    st.sidebar.write(f"Average Study Time:{df['StudyTimeWeekly'].mean():.2f} hrs")
    st.sidebar.write(f"Average Absences:{df['Absences'].mean():.2f}")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Grade Analysis", "Attendance Analysis", "Model Prediction"])
    # Tab 1: Overview
    with tab1:
        st.subheader("Dataset Overview")
        st.write(df.describe(include='all'))
        st.write("Data types:")
        st.write(df.dtypes)
        st.write("Missing values:")
        st.write(df.isnull().sum())

    # Tab 2: = Grade Analysis
    with tab2:
        st.subheader("Student Performance Summary")

        # Pass/Fail classification based on GPA
        df['Pass'] = df['GPA'].apply(lambda x: "Pass" if x >= 2.0 else "Fail")

        # Pie chart of pass/fail
        pass_counts = df['Pass'].value_counts()
        fig, ax = plt.subplots()
        ax.pie(pass_counts, labels=pass_counts.index, autopct='%1.1f%%', startangle=90, colors=['#4CAF50','#F44336'])
        ax.set_title("Pass vs Fail Distribution")
        st.pyplot(fig)
        # Bar chart of average GPA by grade
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="GradeClass", y="GPA", data=df, ax=ax, estimator="mean", palette="Blues_d")
        ax.set_title("Average GPA by Grade")
        st.pyplot(fig)
        # Highlight weak students
        st.write("At-Risk Students (GPA < 2.0 or Absences > 20)")
        at_risk = df[(df['GPA'] < 2.0) | (df['Absences'] > 20)]
        st.dataframe(at_risk)
    # Tab 3: Attendance Insights
    with tab3:
        st.subheader("Attendance Analysis")

        # Absences by grade
        fig, ax = plt.subplots(figsize=(8,4))
        sns.barplot(x="GradeClass", y="Absences", data=df, ax=ax, estimator="mean")
        ax.set_title("Average Absences by Grade")
        st.pyplot(fig)

        # Absences vs GPA
        st.write("GPA vs Absences (Correlation)")
        fig, ax = plt.subplots(figsize=(6,4))
        sns.scatterplot(x="Absences", y="GPA", data=df, hue="Pass", ax=ax)
        st.pyplot(fig)

    # Tab 4: Predictive Modeling
    with tab4:
        st.subheader("Predict Pass/Fail with Different ML Algorithms")

        # Convert pass/fail to binary
        df['Pass_binary'] = df['GPA'].apply(lambda x: 1 if x >= 2.0 else 0)
        X = df[["GPA","StudyTimeWeekly","GradeClass","Absences"]]
        y = df["Pass_binary"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Choose algorithm
        algo_choice = st.selectbox("Choose an ML Algorithm",["Logistic Regression", "Decision Tree", "Random Forest"])
        # Initialize model
        if algo_choice == "Logistic Regression":
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression()
        elif algo_choice == "Decision Tree":
            from sklearn.tree import DecisionTreeClassifier
            model = DecisionTreeClassifier()
        else: # Random Forest
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier()

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Show accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"{algo_choice} Accuracy:{accuracy:.2f}")

        # predction 
        st.subheader("Predict for a Student")
        GPA_input = st.number_input("Enter GPA (0.0 - 4.0)", min_value=0.0, max_value=4.0, step=0.1)
        StudyTimeWeekly_input = st.number_input("Enter Study Time Weekly (hours)", min_value=0, max_value=100, step=1)
        Gradeclass_input = st.number_input("Enter Grade Class (1-4)", min_value=1, max_value=4, step=1)
        Absences_input = st.number_input("Enter Number of Absences", min_value=0, max_value=100, step=1)

        if st.button("Predict Outcome"):
            input_data = pd.DataFrame({
                "GPA": [GPA_input],
                "StudyTimeWeekly": [StudyTimeWeekly_input],
                "GradeClass": [Gradeclass_input],
                "Absences": [Absences_input]
            })
            prediction = model.predict(input_data)
            result = "Pass" if prediction[0] == 1 else "Fail"
            st.write(f"The student is predicted to: {result}")

   