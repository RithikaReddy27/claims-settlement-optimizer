# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib, json

df = pd.read_csv("synthetic_claims_settlement_data.csv")

# Encode target
action_map = {"Litigate": 0, "Mediate": 1, "Settle": 2}
df["Recommended_Action"] = df["Recommended_Action"].map(action_map)

id_columns = ["Claim ID", "Customer ID", "Policy Number"]
df.drop(columns=[col for col in id_columns if col in df.columns], inplace=True, errors='ignore')

X = df.drop("Recommended_Action", axis=1)
y = df["Recommended_Action"]

X_encoded = pd.get_dummies(X)
with open("feature_columns.json", "w") as f:
    json.dump(X_encoded.columns.tolist(), f)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, "settlement_strategy_model.pkl")

print("✅ Model trained and saved.")
print("📊 Classification Report:")
print(classification_report(y_test, model.predict(X_test)))
