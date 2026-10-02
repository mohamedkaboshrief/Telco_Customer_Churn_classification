# Telco_Customer_Churn_classification

This project focuses on predicting whether a telecom customer is likely to churn using machine learning classification models.

Recall is prioritized because identifying customers who are likely to churn is important, and missing actual churners may be costly.

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
* Bivariate analysis between numerical features and `Churn`.
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
* Set `TotalCharges` to 0 for customers with blank charges.
* Set `tenure` to 0 when `TotalCharges` was 0.
* Estimated missing `tenure` values using `TotalCharges` and `MonthlyCharges`.
* Estimated missing `MonthlyCharges` using `TotalCharges` and `tenure`.
* Replaced `No internet service` with `No` for the relevant service features.
* Replaced `No phone service` with `No` in `MultipleLines`.
* Removed duplicate rows.

During EDA, `gender` showed limited impact on churn, so it was removed from the final feature set to simplify the model.

---

### 3. Feature Preprocessing

---

Features were divided into three groups.

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

A stratified split was used to preserve the class distribution of `Churn` across the training and test sets.

```python
train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=44,
    shuffle=True,
    stratify=y
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

Logistic Regression was selected as the final model after evaluating the tested models, with Recall prioritized for the Churn class.

The final test-set results were:

| Metric          |  Score |
| --------------- | -----: |
| Accuracy        | 70.66% |
| Churn Precision | 47.01% |
| Churn Recall    | 86.42% |
| Churn F1-Score  | 60.90% |
| ROC-AUC         | 84.20% |

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

Different classification thresholds were tested to examine the trade-off between Precision, Recall, and F1-score.

| Threshold | Precision |    Recall |  F1-Score |
| --------- | --------: | --------: | --------: |
| 0.3       |     37.1% |     95.3% |     53.4% |
| 0.4       |     41.1% |     92.7% |     57.0% |
| **0.5**   | **47.0%** | **86.4%** | **60.9%** |
| 0.6       |     56.9% |     71.8% |     63.5% |
| 0.7       |     67.7% |     53.2% |     59.6% |

A threshold of **0.5** was retained for the final model because Recall is the primary metric in this project. Although a threshold of 0.6 produced a higher F1-score, it reduced Recall from **86.4% to 71.8%**.

---

## ROC-AUC

---

The final Logistic Regression model achieved a **ROC-AUC of 0.842**.

The ROC curve was plotted using the predicted probabilities to evaluate the model's ability to distinguish between churn and non-churn customers across different classification thresholds.

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

The application also handles dependent input features:

* When `InternetService` is `No`, internet-related services are automatically set to `No`.
* When `PhoneService` is `No`, `MultipleLines` is automatically set to `No`.

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
* Stratified Splitting
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

The final Logistic Regression model achieved **86.42% Recall**, **47.01% Precision**, **60.90% F1-Score**, and **0.842 ROC-AUC** on the test set.

A classification threshold of **0.5** was retained because Recall was prioritized, allowing the model to identify a larger proportion of actual churners.

The trained model was saved as a complete pipeline and deployed through a Streamlit web application for real-time churn prediction.

---

## Author

---

**Mohamed Aboshrief**

Machine Learning Engineer Aspirant

[GitHub](https://github.com/mohamedkaboshrief)
