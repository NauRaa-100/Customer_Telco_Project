

#  Telco Customer Churn Prediction

A complete **Machine Learning pipeline** for predicting customer churn using the **Telco Customer Churn dataset** (Kaggle).  
The project includes full preprocessing, feature selection, model comparison (including XGBoost), and performance visualization.

---

##  Project Overview

This project aims to predict whether a telecom customer is likely to **churn (leave the service)** based on demographic and usage data.  
By understanding key churn drivers, telecom companies can improve retention and customer experience.

---

##  Workflow Summary

###  1. Data Loading & Cleaning
- Loaded Telco dataset (`telc.csv`)
- Converted `TotalCharges` from string to numeric
- Handled missing values and incorrect data types
- Encoded the target variable `Churn` as binary (0 = No, 1 = Yes)

###  2. Exploratory Data Analysis (EDA)
- Visualized churn distribution  
- Examined relationships with categorical features (e.g., gender, contract type)
- Correlation heatmap to identify strong predictors

###  3. Feature Engineering
- Used **One-Hot Encoding** for categorical variables  
- Scaled numeric features using **StandardScaler**  
- Selected top 20 important features using **Random Forest feature importances**

###  4. Model Training & Evaluation
Trained and compared multiple algorithms:
- **Logistic Regression**
- **Decision Tree**
- **Random Forest**
- **Support Vector Machine (SVM)**
- **XGBoost** *(if available)*

Each model was evaluated using:
- Accuracy  
- Precision  
- Recall  
- F1 Score  
- AUC (Area Under the ROC Curve)

###  5. Visualization
- Churn distribution plots  
- Confusion Matrix for best model  
- ROC Curve visualization  
- Feature importance bar chart  

###  6. Optimization & Saving
- Optimized memory usage by converting data types  
- Saved the trained scaler and best model with **Joblib**

---

##  Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|-----------|-----------|--------|----|-----|
| Logistic Regression | ~79% | - | - | - | - |
| Decision Tree | ~81% | - | - | - | - |
| Random Forest | ~85% | - | - | - | - |
| **XGBoost** | **87%** | - | - | - | **Best** |

> XGBoost achieved the highest AUC and overall performance.

---

##  Tech Stack

| Library | Purpose |
|----------|----------|
| **Python 3.x** | Core language |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Scikit-learn** | ML models & metrics |
| **XGBoost** | Gradient boosting |
| **Joblib** | Model persistence |

---

##  How to Run

 **Clone this repository**
   ```bash
   git clone https://github.com/your-username/telco-customer-churn.git
   cd telco-customer-churn
````


---

##  Insights

* Customers on **month-to-month contracts** had the highest churn rates.
* **Electronic check payments** correlated strongly with churn.
* **Longer tenure** and **automatic payments** were indicators of retention.

---

##  Saved Outputs

| File                     | Description                       |
| ------------------------ | --------------------------------- |
| `scaler.pkl`             | Fitted StandardScaler             |
| `best_model_XGBoost.pkl` | Best performing model             |
| `results_df.csv`         | Evaluation metrics for all models |

---

##  Future Improvements

* Implement **GridSearchCV** for hyperparameter tuning
* Add **cross-validation** for more robust evaluation
* Deploy model via **Streamlit** or **Flask API**
* Add explainability via **SHAP values**

---

##  Author

**Nau Raa**
Data Science & Machine Learning Enthusiast


 *If you find this useful, please star the repository on GitHub!* 
