import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("student.csv")
print(df)
print(df.head())
#Bar chart
plt.bar(df["StudentID"], df["GPA"], color= "red")
plt.title("GPA of students")
plt.xlabel("Students")
plt.ylabel("GPA")
plt.show()
#Pie chart
grade_count = df["Gradeclass"].value_counts()
plt.pie(grade_count, labels= grade_count.index, autopct="%1.1f%%", startangle=90)
plt.title("Grade Class Distribution")
plt.show()
#line chart
plt.plot(df["StudentID"], df["GPA"], marker='o', linestyle='-', color='green')
plt.title("GPA Trend")
plt.xlabel("Students")
plt.ylabel("GPA")
plt.show()
#scatter plot
plt.scatter(df["StudentID"], df["GPA"], color='blue')
plt.title("GPA Scatter Plot")
plt.xlabel("Students")
plt.ylabel("GPA")
plt.show()
#Histogram
plt.hist(df["GPA"], bins=5, color='purple', edgecolor='black')
plt.title("GPA Distribution")
plt.xlabel("GPA")
plt.ylabel("no of Students")
plt.show()
