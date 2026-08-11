# 🎓 Student Pass Prediction using Logistic Regression

A machine learning project that predicts whether a student is likely to **pass or fail** based on academic, attendance, study, and personal factors.

The project uses **Logistic Regression**, a supervised machine learning algorithm commonly used for binary classification.

---

## 📌 Project Overview

The goal of this project is to build a model that predicts:

* `1` → Student Passes
* `0` → Student Fails

The model uses information such as:

* Hours studied
* Attendance
* Previous scores
* Sleep hours
* Tutoring sessions
* Physical activity
* Motivation level
* Internet access
* Family income
* Teacher quality
* And other student-related features

The project helped me understand the complete workflow of a binary classification problem.

---

## 🧠 Machine Learning Algorithm

### Logistic Regression

Logistic Regression is used because our target variable has two possible outcomes:

```text
0 → Fail
1 → Pass
```

Instead of directly predicting a continuous value, Logistic Regression estimates the **probability** of belonging to a class.

For example:

```text
Probability of passing = 0.87
```

The model can interpret this as an approximately **87% probability of passing**.

Using the default classification threshold:

```text
Probability ≥ 0.5 → Pass
Probability < 0.5 → Fail
```

---

## 📊 Dataset

The dataset contains information about students and their academic performance.

The original dataset includes features related to:

* Study habits
* Attendance
* Previous academic performance
* Sleep
* Tutoring
* Physical activity
* Motivation
* Family background
* School-related factors

The original `Exam_Score` column was used to create the target variable:

```python
df["Pass"] = (df["Exam_Score"] >= 60).astype(int)
```

Therefore:

```text
Exam Score >= 60 → Pass (1)
Exam Score < 60  → Fail (0)
```

`Exam_Score` was then excluded from the model's input features to avoid **data leakage**, since it was directly used to create the target.

---

## 🔧 Project Workflow

The project follows a typical machine learning workflow:

```text
Dataset
   ↓
Data Exploration
   ↓
Missing Value Handling
   ↓
Target Creation
   ↓
Categorical Data Encoding
   ↓
Exploratory Data Analysis
   ↓
Feature/Target Separation
   ↓
Train-Test Split
   ↓
Logistic Regression
   ↓
Predictions
   ↓
Model Evaluation
```

---

## 🧹 Data Preprocessing

### 1. Handling Missing Values

Missing values were identified using:

```python
df.isnull().sum()
```

The missing values were handled before training the model.

### 2. Encoding Categorical Variables

Categorical variables were converted into numerical variables using one-hot encoding:

```python
df = pd.get_dummies(
    df,
    columns=categorical_columns,
    drop_first=True,
    dtype=int
)
```

This allowed the Logistic Regression model to work with categorical information.

---

## 📈 Exploratory Data Analysis

Several visualizations were used to understand the dataset and identify patterns.

### Pass vs Fail Distribution

A count plot was used to understand the distribution of students between the two classes.

### Hours Studied vs Pass/Fail

A box plot was used to compare study hours between students who passed and failed.

### Attendance vs Pass/Fail

A box plot was used to investigate the relationship between attendance and passing.

### Previous Scores vs Pass/Fail

A box plot was used to compare previous academic performance between the two groups.

### Correlation Heatmap

A correlation heatmap was used to examine relationships between numerical variables.

---

## 🧪 Train-Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

The data was divided into:

```text
80% → Training data
20% → Testing data
```

The model learns patterns from the training data and is evaluated on the unseen testing data.

---

## 🤖 Model Training

The Logistic Regression model was created using Scikit-learn:

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
```

The model learns relationships between the student features and the Pass/Fail target.

---

## 🎯 Making Predictions

Predictions were generated using:

```python
y_pred = model.predict(X_test)
```

The model can also provide probabilities:

```python
y_prob = model.predict_proba(X_test)
```

For example:

```text
Fail probability = 0.12
Pass probability = 0.88
```

This means the model estimates an 88% probability that the student will pass.

---

## 📊 Model Evaluation

The model was evaluated using several classification metrics.

### Accuracy

Measures the percentage of all predictions that were correct.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Answers:

> When the model predicts Pass, how often is it correct?

### Recall

Answers:

> Of all students who actually passed, how many did the model correctly identify?

### F1 Score

Combines Precision and Recall into a single score and is useful when both are important.

### Confusion Matrix

The confusion matrix shows:

```text
True Positive
True Negative
False Positive
False Negative
```

A classification report was also generated using:

```python
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

---

## 💡 What I Learned

Through this project, I learned how to:

* Work with a real-world dataset
* Inspect and handle missing values
* Create a binary target variable
* Encode categorical variables
* Perform exploratory data analysis
* Separate features and targets
* Split data into training and testing sets
* Train a Logistic Regression model
* Generate class predictions
* Generate prediction probabilities
* Understand confusion matrices
* Evaluate classification models using Accuracy, Precision, Recall, and F1 Score
* Understand the importance of avoiding data leakage

---

## 🚀 Future Improvements

Some possible improvements for this project include:

* Testing different classification algorithms
* Comparing Logistic Regression with Decision Trees and Random Forest
* Hyperparameter tuning
* Feature selection
* Cross-validation
* Improving the handling of class imbalance
* Creating a simple web interface for making predictions
* Deploying the model as a small ML application

---

## 👨‍💻 Author

**Aarush Archit**

Student at Delhi Technological University (DTU)

This project was created as part of my journey to learn Machine Learning and build practical ML projects.
