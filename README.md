# Telco_Customer_Churn_classification

A machine learning classification project focused on predicting customer churn using customer demographic, service, contract, and billing information.

The project covers an end-to-end machine learning workflow, including data cleaning, missing-value handling, feature encoding, feature scaling, model comparison, cross-validation, hyperparameter tuning, and model evaluation.

## Dataset

The dataset contains **7,043 customer records and 21 columns** before preprocessing.

**Source:** [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/jethwaaatmik/telco-customer-churn-dataset)

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

The dataset contains a large number of missing values in several columns, so handling them carefully was a key part of the project:

| Column           | Missing values |
| ---------------- | -------------: |
| `tenure`         |          2,500 |
| `MonthlyCharges` |          1,500 |
| `StreamingTV`    |          1,500 |
| `Partner`        |          1,000 |
| `InternetService`|          1,000 |
| `gender`         |            750 |
| `TotalCharges`   |     11 (blank) |

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

* Converting `TotalCharges` to a numerical data type (blank values were treated as missing)
* Merging redundant categories such as `No internet service` and `No phone service` into `No`
* Filling missing `InternetService` and `StreamingTV` values using related service columns and the most frequent value within each group
* Marking missing `gender` and `Partner` values as `unknown`
* Estimating missing numerical values using related features:
  * `TotalCharges` from `MonthlyCharges × tenure`
  * `tenure` from `TotalCharges ÷ MonthlyCharges`
  * `MonthlyCharges` from the median of similar customers (same services and contract)
* Removing the 3 rows that could not be recovered

After preprocessing, the dataset contained **7,040 records and 20 columns** before encoding.

### 3. Feature Encoding

Different encoding techniques were used depending on the feature type:

* **Label Encoding** for binary categorical features
* **One-Hot Encoding** for nominal categorical features (`gender`, `Partner`, `Dependents`, `InternetService`, `PaymentMethod`)
* **Ordinal Encoding** for `Contract`, preserving its natural order:

```text
Month-to-month → 0
One year       → 1
Two year       → 2
```

After encoding, the dataset contained **29 features** (30 columns including the target).

### 4. Outlier Analysis

Box plots were used to inspect potential outliers in numerical features, including:

* `MonthlyCharges`
* `TotalCharges`
* `tenure`

### 5. Model Training

The processed data was split into training and testing sets using an **80/20 split** (5,632 training records and 1,408 test records).

The features were standardized using `StandardScaler` (fitted on the training set only) for the scale-sensitive models: KNN, SVC, Logistic Regression, and SGD.

Because the main objective was to identify customers who are likely to churn, **Recall** for the churn class was selected as the primary metric.

## Model Comparison

Seven classification models were tuned using `GridSearchCV` with **5-Fold Cross-Validation**, scored by **Recall**:

| Model                | Best CV Recall |
| -------------------- | -------------: |
| XGBoost              |          0.510 |
| Random Forest        |          0.482 |
| Gaussian Naive Bayes |          0.745 |
| K-Nearest Neighbors  |          0.542 |
| SVC                  |          0.824 |
| Logistic Regression  |          0.802 |
| SGD Classifier       |      **0.827** |

`class_weight='balanced'` was selected by the grid search for Logistic Regression, SVC, and the SGD Classifier, which helps with the class imbalance (about 27% of the test customers churned).

## Final Model Selection

The three models with the highest cross-validation recall (SGD, SVC, and Logistic Regression) were evaluated once on the held-out test set:

| Model               | Accuracy | Churn Precision | Churn Recall | Churn F1-Score |
| ------------------- | -------: | --------------: | -----------: | -------------: |
| **Logistic Regression** |   **74.1%** |         **51%** |      **84%** |        **64%** |
| SVC (linear)        |    70.9% |             48% |          85% |            61% |
| SGD Classifier      |    70.0% |             47% |          86% |            61% |

The three models reach a similar recall, but **Logistic Regression** gave the best balance between recall, precision, and F1-score, so it was selected as the final model.

**Best Hyperparameters (Logistic Regression):**

```text
C            = 0.01
class_weight = balanced
solver       = liblinear
```

**Best Cross-Validation Recall:**

```text
0.8019
```

## Final Model Results

The tuned Logistic Regression model achieved the following results on the test set:

| Metric          |   Score |
| --------------- | ------: |
| Train Accuracy  |  73.54% |
| Test Accuracy   |  74.08% |
| Churn Precision |     51% |
| Churn Recall    | **84%** |
| Churn F1-Score  |     64% |

### Confusion Matrix

|              | Predicted 0 | Predicted 1 |
| ------------ | ----------: | ----------: |
| **Actual 0** |         723 |         305 |
| **Actual 1** |          60 |         320 |

The model correctly identified **320 out of 380 actual churn cases**, resulting in approximately **84% recall for the churn class**.

The model's lower precision for the churn class (51%) indicates that it also flags a number of customers as potential churners who did not actually churn. This is an expected trade-off when the goal is to miss as few churners as possible.

## Limitations and Future Work

* Move the imputation and scaling steps into a scikit-learn `Pipeline`, so they are fitted only on the training folds.
* Add ROC-AUC / PR-AUC and tune the decision threshold to balance recall and precision.
* Apply class weighting (`class_weight` / `scale_pos_weight`) to the tree-based models for a fairer comparison.
* Add exploratory visualizations and analyze the model coefficients to identify the main drivers of churn.
* Save the trained model (for example with `joblib`) so it can be reused for predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Jupyter Notebook

## Machine Learning Concepts

* Exploratory Data Analysis
* Missing Value Handling
* Data Preprocessing
* Feature Encoding
* Feature Scaling
* Outlier Analysis
* Train/Test Splitting
* Cross-Validation
* Model Comparison
* Hyperparameter Tuning
* GridSearchCV
* Class Weighting
* Classification Metrics
* Confusion Matrix

## How to Run

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost jupyter
```

Place `telco_data.csv` in the same folder as the notebook, then open and run `Telco_Customer_Churn.ipynb`.

## Project Structure

```text
Telco_Customer_Churn_classification/
│
├── Telco_Customer_Churn.ipynb
├── README.md
└── .gitignore
```

## Conclusion

This project demonstrates an end-to-end machine learning workflow for customer churn prediction.

After preprocessing a dataset with substantial missing values and comparing seven classification algorithms using 5-Fold Cross-Validation, the linear models achieved the highest recall. **Logistic Regression** was selected as the final model because it offered the best balance between recall and precision.

The final model achieved approximately **74% test accuracy** and **84% recall for the churn class**, allowing it to identify most of the customers who actually churned.

## Author

**Mohamed Khaled Elsayed Ahmed Aboshrief**

Computer Science & Engineering Student
