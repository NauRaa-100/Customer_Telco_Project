"""
-- Telco Customer Churn structured dataset (Kaggle)

"""

# === Customer Churn - Corrected Pipeline with XGBoost ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, classification_report, confusion_matrix)
import joblib

# try import xgboost (if not installed, pip install xgboost)
try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False
    print("XGBoost not available. Install with: pip install xgboost")



# ---------- Load ----------
df = pd.read_csv(r'E:\Rev-DataScience\AI-ML\MLProjects_Structured_Data\telc.csv', encoding='latin')
print(df.shape)
print(df.describe())
print(df.info())
# ---------- Basic cleanup ----------
# map target
df['Churn'] = df['Churn'].map({'No':0, 'Yes':1})

# If TotalCharges present as string, convert (Telco dataset quirk)
if 'TotalCharges' in df.columns:
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges'])

# ---------- EDA quick ----------
print(df.groupby('gender')['Churn'].mean())
print(df.loc[df['Churn']=='Yes','gender'].value_counts())
print(df['Churn'].value_counts())
'''
plt.figure(figsize=(8,4))
sns.countplot(x='Churn', data=df)
plt.title('Churn distribution (0=No, 1=Yes)')
plt.show()
'''
# ---------- Encode categoricals ----------
cat_cols = df.select_dtypes(include=['object']).columns.tolist()
# exclude columns that are already target-coded
cat_cols = [c for c in cat_cols if c != 'Churn']
df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)

# ---------- Train/Test split (important: split BEFORE feature selection) ----------
X = df_enc.drop('Churn', axis=1)
y = df_enc['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ---------- Scaling (fit scaler on train only) ----------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- Feature importance on train (RandomForest) ----------

rf_temp = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_temp.fit(X_train_scaled, y_train)
importances = pd.Series(rf_temp.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Top features:\n", importances.head(10))

# Choose top k features (example: 20)
top_k = 20
top_features = importances.head(top_k).index.tolist()

# Reduce datasets to top features
X_train_sel = pd.DataFrame(X_train_scaled, columns=X.columns)[top_features].values
X_test_sel = pd.DataFrame(X_test_scaled, columns=X.columns)[top_features].values

# ---------- Models to try (include XGBoost if available) ----------
models = {
    'Logistic': LogisticRegression(max_iter=500, random_state=42),
    'DecisionTree': DecisionTreeClassifier(random_state=42, max_depth=6),
    'RandomForest': RandomForestClassifier(random_state=42, n_estimators=200),
}
if xgb_available:
    models['XGBoost'] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
models['SVM'] = SVC(kernel='rbf', probability=True, random_state=42)

results = []
for name, model in models.items():
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    y_prob = model.predict_proba(X_test_sel)[:,1]
    
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'AUC': roc_auc_score(y_test, y_prob)
    })

results_df = pd.DataFrame(results).sort_values(by='AUC', ascending=False)
print(results_df)

# ---------- Pick best (by AUC) and show detailed eval ----------
best_name = results_df.iloc[0]['Model']
best_model = models[best_name]
print("Best model:", best_name)

y_pred = best_model.predict(X_test_sel)
y_prob = best_model.predict_proba(X_test_sel)[:,1]

print(classification_report(y_test, y_pred))

# Confusion matrix (plot)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
#plt.title(f'Confusion Matrix - {best_name}')
#plt.xlabel('Predicted')
#plt.ylabel('Actual')
#plt.show()

#---------Seperate----------
'''
# ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
auc = roc_auc_score(y_test, y_prob)
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
plt.plot([0,1],[0,1],'k--')
plt.title(f'ROC Curve - {best_name}')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()
'''
#---------Seperate----------
'''
corr = df_enc.corr()['Churn'].sort_values(ascending=False)
sns.heatmap(corr,annot=True,cmap='coolwarm')
plt.title('Top Features Correlation')
plt.show()

'''
# ---------- Optimization ----------
print('Before =>\n',df.memory_usage(deep=True))

for col in df.columns:
    if df[col].dtype == 'object':
        df[col]=df[col].astype('category')

for col in df.columns:
    if df[col].dtype == 'int64':
        df[col]=df[col].astype('int16')

for col in df.columns:
    if df[col].dtype== 'float64':
        df[col]=df[col].astype('float16')

print('After => \n',df.memory_usage(deep=True))



# ---------- Save scaler + best model ----------

joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_model, f'best_model_{best_name}.pkl')
print("Saved scaler and model.")

# Feature importance plot from rf_temp
plt.figure(figsize=(8,6))
sns.barplot(x=importances.values[:15], y=importances.index[:15])
plt.title("Top 15 Feature Importances (RandomForest on train)")
plt.show()
