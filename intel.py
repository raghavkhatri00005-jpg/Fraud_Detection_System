# Financial Fraud Detection
# Install: pip install pandas numpy scikit-learn matplotlib seaborn

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

os.makedirs("output", exist_ok=True)

# 1. Load / Create Dataset
if os.path.exists("creditcard.csv"):
    df = pd.read_csv("creditcard.csv")
else:
    np.random.seed(42)
    n = 50000
    fraud = np.random.rand(n) < 0.01
    df = pd.DataFrame({
        "amount": np.where(fraud, np.random.gamma(5,100,n), np.random.gamma(2,30,n)),
        "txn_count": np.where(fraud, np.random.poisson(8,n), np.random.poisson(2,n)),
        "distance": np.where(fraud, np.random.exponential(300,n), np.random.exponential(15,n)),
        "night": np.where(fraud, np.random.rand(n)<0.7, np.random.rand(n)<0.1).astype(int),
        "Class": fraud.astype(int)
    })

# 2. Clean Data
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))

print("Dataset:", df.shape)
print("Fraud %:", round(df["Class"].mean()*100, 2))

# 3. EDA
sns.countplot(x="Class", data=df)
plt.title("Genuine vs Fraud")
plt.savefig("output/class_distribution.png")
plt.close()

sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("output/correlation.png")
plt.close()

# 4. Prepare Data
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Train Models
models = {
    "Logistic Regression": LogisticRegression(class_weight="balanced", max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42
    )
}

for name, model in models.items():
    start = time.time()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    print(f"\n{name}")
    print("Precision:", round(precision_score(y_test, pred), 3))
    print("Recall:", round(recall_score(y_test, pred), 3))
    print("F1 Score:", round(f1_score(y_test, pred), 3))
    print("Time:", round(time.time()-start, 2), "sec")

# 6. Final Prediction with Random Forest
rf = models["Random Forest"]
prob = rf.predict_proba(X_test)[:, 1]
final_pred = (prob >= 0.8).astype(int)

print("\nFinal Result - Random Forest")
print(classification_report(y_test, final_pred))

tn, fp, fn, tp = confusion_matrix(y_test, final_pred).ravel()
print("Fraud Caught:", tp)
print("Fraud Missed:", fn)
print("False Positives:", fp)

# 7. Confusion Matrix
sns.heatmap(confusion_matrix(y_test, final_pred),
            annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("output/confusion_matrix.png")
plt.close()

# 8. Feature Importance
pd.Series(rf.feature_importances_, index=X.columns).sort_values().plot(
    kind="barh", title="Feature Importance"
)
plt.tight_layout()
plt.savefig("output/feature_importance.png")
plt.close()

print("\nDone! Check the 'output' folder.")