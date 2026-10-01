# Telco_Customer_Churn_classification

This project focuses on predicting whether a telecom customer is likely to churn using machine learning classification models.

Recall is prioritized because identifying customers who are likely to churn is more important than missing actual churners.

---

## Live Demo

---

[Try the Streamlit App](https://telcocustomerchurnclassification-123.streamlit.app/)

---

## Dataset

---

The dataset contains **7,043 customers** and **21 features**.

* `Churn` is the target variable.
* `Yes` → 1
* `No` → 0
* `customerID` was removed because it is only an identifier.

Dataset source:

[Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/jethwaaatmik/telco-customer-churn-dataset)

---

## Project Workflow

---

### 1. Exploratory Data Analysis

---

EDA was performed to understand the distribution of the features and their relationship with customer churn.

The analysis included:

* Numerical feature distributions using histograms and boxplots.
* Categorical feature analysis using countplots.
* Bivariate analysis between features and `Churn`.
* Churn class distribution.

Some observations from the analysis:

* Customers with shorter tenure tend to churn more frequently.
* Customers with higher monthly charges tend to churn more frequently.
* Contract type, internet service, and payment method show noticeable differences in churn behavior.

---

### 2. Data Cleaning

---

The following cleaning steps were performed:

* Removed `customerID`.
* Converted `TotalCharges` to numeric.
* Handled blank values in `TotalCharges`.
* Adjusted `tenure` for customers with zero `TotalCharges`.
* Estimated missing `tenure` and `MonthlyCharges` values where possible.
* Replaced values such as `No internet service` and `No phone service` with `No`.
* Removed duplicate rows.

During EDA, `gender` showed limited impact on churn, so it was removed from the final feature set to simplify the model.

---

### 3. Feature Preprocessing

---

Features were divided into:

**Binary Features:**

* Partner
* Dependents
* PhoneService
* MultipleLines
* OnlineSecurity
* OnlineBackup
* DeviceProtection
* TechSupport
* StreamingTV
* StreamingMovies
* PaperlessBilling

**Categorical Features:**

* InternetService
* Contract
* PaymentMethod

**Numerical Features:**

* tenure
* MonthlyCharges
* TotalCharges

The preprocessing pipeline includes:

* `SimpleImputer`
* `StandardScaler`
* `OrdinalEncoder`
* `OneHotEncoder`
* `ColumnTransformer`

---

### 4. Feature Selection

---

`SelectKBest` with `f_classif` was used for feature selection.

Different values of `k` were tested during hyperparameter tuning:

```text
5, 10, 15, 20, all
```

Feature selection was included inside the pipeline to keep preprocessing and feature selection consistent during cross-validation.

---

## Train/Test Split

---

The dataset was split into:

* **75% Training Data**
* **25% Test Data**

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=44,
    shuffle=True
)
```

The target variable was mapped as:

```text
Yes → 1
No → 0
```

---

## Model Training

---

Five classification models were trained and tuned using `GridSearchCV`:

* Logistic Regression
* Support Vector Classifier (SVC)
* Random Forest
* K-Nearest Neighbors (KNN)
* Gradient Boosting

Each model was implemented using a Pipeline containing preprocessing, feature selection, and the classification model.

---

## Model Comparison

---

The models were tuned using **5-fold cross-validation** with **Recall** as the scoring metric.

Logistic Regression and SVC achieved the highest cross-validation recall among the tested models.

Both models were then evaluated on the test set using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Final Model Selection

---

Based on the evaluation results, **Logistic Regression** was selected as the final model.

The model achieved approximately:

| Metric          | Score |
| --------------- | ----: |
| Accuracy        |   74% |
| Churn Precision |   49% |
| Churn Recall    |   86% |
| Churn F1-Score  |   63% |

Recall was prioritized because a False Negative represents a customer who actually churns but was not identified by the model.

---

## Confusion Matrix

---

A confusion matrix was used to analyze the model's predictions and understand:

* True Positives
* True Negatives
* False Positives
* False Negatives

This is especially important for churn prediction because False Negatives represent missed churners.

---

## Threshold Analysis

---

Different classification thresholds were tested to examine the trade-off between Precision and Recall.

| Threshold | Precision |    Recall |  F1-Score |
| --------- | --------: | --------: | --------: |
| 0.3       |     37.8% |     96.1% |     54.3% |
| 0.4       |     42.1% |     92.4% |     57.9% |
| **0.5**   | **49.1%** | **85.6%** | **62.4%** |
| 0.6       |     55.9% |     67.9% |     61.3% |
| 0.7       |     64.4% |     48.4% |     55.2% |

Among the tested thresholds, **0.5 achieved the highest F1-score**, so the default threshold of 0.5 was retained.

---

## ROC-AUC

---

ROC-AUC was calculated using the predicted probabilities of the final Logistic Regression model.

The ROC curve was also plotted to evaluate the model's ability to distinguish between churn and non-churn customers across different thresholds.

---

## Model Saving

---

The complete trained pipeline was saved using `joblib`:

```python
import joblib

final_model = logistic_grid.best_estimator_

joblib.dump(final_model, "telco_churn_model.pkl")
```

The saved pipeline contains the preprocessing, feature selection, and Logistic Regression model.

---

## Streamlit Application

---

A Streamlit application was created to allow users to enter customer information and receive a churn prediction.

The application uses the saved machine learning pipeline to process the input and predict whether the customer is likely to churn.

**Live Application:**

[Open Telco Churn Prediction App](https://telcocustomerchurnclassification-123.streamlit.app/)

---

## Technologies Used

---

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit
* Jupyter Notebook

---

## Machine Learning Concepts

---

This project demonstrates:

* Exploratory Data Analysis
* Data Cleaning
* Feature Preprocessing
* Feature Selection
* Train/Test Split
* Pipelines
* ColumnTransformer
* Cross-Validation
* GridSearchCV
* Logistic Regression
* SVC
* Random Forest
* KNN
* Gradient Boosting
* Classification Metrics
* Confusion Matrix
* Threshold Analysis
* ROC-AUC
* Model Serialization
* Streamlit Deployment

---

## How to Run

---

Clone the repository:

```bash
git clone https://github.com/mohamedkaboshrief/Telco_Customer_Churn_classification.git
```

Install the required packages:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Project Structure

---

```text
Telco_Customer_Churn_classification/
│
├── README.md
├── Telco_Customer_Churn.ipynb
├── app.py
├── telco_churn_model.pkl
├── requirements.txt
├── telco_data.csv
└── .gitignore
```

---

## Conclusion

---

This project demonstrates an end-to-end machine learning workflow for customer churn prediction, from data exploration and cleaning to preprocessing, feature selection, model tuning, evaluation, and deployment.

The final model was deployed as a Streamlit web application for real-time churn prediction.

---

## Author

---

**Mohamed Aboshrief**

Machine Learning Engineer Aspirant

[GitHub](https://github.com/mohamedkaboshrief)
