# Telco_Customer_Churn_classification

A machine learning classification project focused on predicting customer churn using customer demographic, service, contract, and billing information.

The project covers an end-to-end machine learning workflow, including exploratory data analysis, data preprocessing, feature encoding, model comparison, cross-validation, hyperparameter tuning, and model evaluation.

## Dataset

The dataset contains **7,043 customer records and 21 columns** before preprocessing.

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/jethwaaatmik/telco-customer-churn-dataset)

The target variable is:

- `Churn = 0` → Customer does not churn
- `Churn = 1` → Customer churns

The target variable is:

* `Churn = 0` → Customer does not churn
* `Churn = 1` → Customer churns

The features include:

* Customer demographics
* Tenure
* Phone and internet services
* Online security and support services
* Streaming services
* Contract type
* Billing information
* Payment methods
* Monthly charges
* Total charges

## Project Workflow

### 1. Exploratory Data Analysis

The dataset was analyzed to understand its structure and identify:

* Data types
* Missing values
* Duplicate customer IDs
* Unique categorical values
* Potential outliers

The `customerID` column was removed because it is an identifier and does not provide useful predictive information.

### 2. Data Preprocessing

The preprocessing stage included:

* Handling missing values
* Converting `TotalCharges` to a numerical data type
* Estimating missing numerical values using related features
* Handling missing categorical values
* Removing the customer identifier

After preprocessing, the dataset contained **7,040 records and 20 columns** before encoding.

### 3. Feature Encoding

Different encoding techniques were used depending on the feature type:

* **Label Encoding** for binary categorical features
* **One-Hot Encoding** for nominal categorical features
* **Ordinal Encoding** for `Contract`, preserving its natural order:

```text
Month-to-month → 0
One year       → 1
Two year       → 2
```

After encoding, the dataset contained **30 features**.

### 4. Outlier Analysis

Box plots were used to inspect potential outliers in numerical features, including:

* `MonthlyCharges`
* `TotalCharges`
* `tenure`

### 5. Model Training

The processed data was split into training and testing sets using an **80/20 split**.

Six classification models were compared using **5-Fold Cross-Validation**.

Because the main objective was to identify customers who are likely to churn, **Recall** was selected as the primary metric for model comparison.

## Model Comparison

| Model                | Mean CV Recall |
| -------------------- | -------------: |
| Decision Tree        |          0.500 |
| Gaussian Naive Bayes |      **0.723** |
| K-Nearest Neighbors  |          0.405 |
| Logistic Regression  |          0.537 |
| Random Forest        |          0.461 |
| SVC                  |          0.000 |

Based on cross-validation recall, **Gaussian Naive Bayes** was selected for further tuning.

## Hyperparameter Tuning

`GridSearchCV` was used to optimize the `var_smoothing` parameter of Gaussian Naive Bayes.

**Best Parameter:**

```text
var_smoothing = 1e-08
```

**Best Cross-Validation Recall:**

```text
0.7448
```

## Final Model Results

The tuned Gaussian Naive Bayes model achieved the following results on the test set:

| Metric          |   Score |
| --------------- | ------: |
| Train Accuracy  |  75.11% |
| Test Accuracy   |  74.93% |
| Churn Precision |     52% |
| Churn Recall    | **79%** |
| Churn F1-Score  |     63% |

### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------: | ----------: |
| **Actual 0** |         754 |         274 |
| **Actual 1** |          79 |         301 |

The model correctly identified **301 out of 380 actual churn cases**, resulting in approximately **79% recall for the churn class**.

The model's relatively lower precision for the churn class indicates that it also classified a number of customers as potential churners who did not actually churn.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Jupyter Notebook

## Machine Learning Concepts

* Exploratory Data Analysis
* Missing Value Handling
* Data Preprocessing
* Feature Encoding
* Outlier Analysis
* Train/Test Splitting
* Cross-Validation
* Model Comparison
* Hyperparameter Tuning
* GridSearchCV
* Classification Metrics
* Confusion Matrix

## Project Structure

```text
customer-churn-prediction/
│
├── customer_churn_prediction.ipynb
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction.

## Author

**Mohamed Khaled Elsayed Ahmed Aboshrief**

Computer Science & Engineering Student

After preprocessing the data and comparing six classification algorithms using 5-Fold Cross-Validation, **Gaussian Naive Bayes** was selected based on recall and tuned using `GridSearchCV`.


The final model achieved approximately **75% test accuracy** and **79% recall for the churn class**, allowing it to identify a substantial proportion of customers who actually churned.
