import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import torch
import joblib

# Memory cleanup
gc.collect()

# Load data
casino_data = pd.read_csv("casino_players_data.csv")

# **Dataset Analysis**
print("Basic dataset information:")
print(casino_data.info())
print("Target variable distribution (Churn):")
print(casino_data["Churn"].value_counts(normalize=True))
print("Feature statistics:")
print(casino_data.describe())

# Convert categorical columns to numerical values
encoder = LabelEncoder()
for col in ["Favorite_Game", "Player_Type"]:
    if col in casino_data.columns:
        casino_data[col] = encoder.fit_transform(casino_data[col])

# Remove highly correlated features
corr_matrix = casino_data.corr()
high_corr_features = set()
threshold_corr = 0.9  # Correlation threshold
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold_corr:
            colname = corr_matrix.columns[i]
            high_corr_features.add(colname)

casino_data = casino_data.drop(columns=high_corr_features)

# Prepare data
X = casino_data.drop(columns=[col for col in ["Churn", "Player_ID"] if col in casino_data.columns])
y = casino_data["Churn"]

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Detect GPU availability for model training
gpu_available = torch.cuda.is_available()
if gpu_available:
    print("GPU available - enabling GPU support for models!")
else:
    print("GPU not available - training on CPU.")

# **Train initial RandomForest model for feature selection**
rf_initial = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
rf_initial.fit(X_train, y_train)


# **Select important features based on RandomForest model**
def select_important_features(model, X, threshold=0.01):
    feature_importances = model.feature_importances_
    important_features = [X.columns[i] for i in range(len(feature_importances)) if feature_importances[i] > threshold]
    return X[important_features]


X_train = select_important_features(rf_initial, X_train)
X_test = X_test[X_train.columns]  # Remove the same features from the test set

models = {
    "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5, class_weight='balanced',
                                           random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(n_estimators=150, max_depth=6, reg_lambda=1.5, reg_alpha=0.5, use_label_encoder=False,
                                 eval_metric='logloss', verbosity=1),
    "LightGBM": lgb.LGBMClassifier(n_estimators=150, max_depth=6, lambda_l1=1.0, lambda_l2=1.0, verbose=1),
    "CatBoost": cb.CatBoostClassifier(n_estimators=150, depth=6, l2_leaf_reg=1.5, verbose=1,
                                      task_type='GPU' if gpu_available else 'CPU'),
    "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=500)
}

metrics_results = []
for model_name, model in models.items():
    print(f"Training model {model_name}...")
    model.fit(X_train, y_train)
    print(f"Model {model_name} training completed.")

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.6).astype(int)

    report = classification_report(y_test, y_pred, output_dict=True)
    accuracy = accuracy_score(y_test, y_pred)

    metrics_results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": report['1']['precision'],
        "Recall": report['1']['recall'],
        "F1-Score": report['1']['f1-score']
    })

    joblib.dump(model, f"{model_name}.pkl")

# Create comparison table
metrics_df = pd.DataFrame(metrics_results)
print(metrics_df)

# Save processed data to disk
casino_data.to_csv("casino_players_data_processed.csv", index=False)
print("Data has been saved to disk.")
