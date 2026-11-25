
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import (
    train_test_split, cross_val_score,
    GridSearchCV, StratifiedKFold, cross_val_predict
)

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc
)

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# 1. Load cleaned dataset
df = pd.read_csv("dataset/cleaned_data.csv")

# 2. Separate input and target
X = df.drop(columns=[
    "label", "attack_type",
    "traffic_type", "difficulty"
])

y = df["label"]

# Remove duplicate feature records
keep = ~X.duplicated()
X = X.loc[keep]
y = y.loc[keep]

# 3. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# 4. Data preprocessing
categorical = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numerical = X.select_dtypes(
    include=["number"]
).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical),
    ("cat", OneHotEncoder(handle_unknown="ignore"),
     categorical)
])

# 5. Define ML models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=50, random_state=42
    )
}

# Hyperparameter settings
parameters = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1, 10]
    },
    "Decision Tree": {
        "classifier__max_depth": [5, 10, 20]
    },
    "Random Forest": {
        "classifier__n_estimators": [50, 100],
        "classifier__max_depth": [10, 20]
    }
}

cv = StratifiedKFold(
    n_splits=5, shuffle=True, random_state=42
)

results = []
cv_results = []
tuning_results = []

best_model = None
best_name = None
best_score = -1

# 6. Train all three models
for name, model in models.items():

    print("\nTraining:", name)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    # 5-fold cross-validation
    scores = cross_val_score(
        pipeline, X_train, y_train,
        cv=cv, scoring="f1", n_jobs=-1
    )

    cv_results.append({
        "Model": name,
        "Mean F1": scores.mean()
    })

    print("Cross-validation F1:", scores.mean())

    # Validation predictions
    predictions = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv, n_jobs=-1
    )

    accuracy = accuracy_score(y_train, predictions)
    precision = precision_score(
        y_train, predictions, zero_division=0
    )
    recall = recall_score(
        y_train, predictions, zero_division=0
    )
    f1 = f1_score(
        y_train, predictions, zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    })

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # Confusion matrix
    cm = confusion_matrix(
        y_train, predictions, labels=[0, 1]
    )

    filename = name.lower().replace(" ", "_")

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Normal", "Malicious"],
        yticklabels=["Normal", "Malicious"]
    )

    plt.title(name + " - Validation Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()

    plt.savefig(
        f"graphs/{filename}_confusion_matrix.png",
        dpi=150
    )

    plt.close()

    # 7. Hyperparameter tuning
    print("Hyperparameter tuning started...")

    grid = GridSearchCV(
        pipeline,
        parameters[name],
        cv=cv,
        scoring="f1",
        n_jobs=-1
    )

    grid.fit(X_train, y_train)

    print("Best parameters:", grid.best_params_)
    print("Best CV F1:", grid.best_score_)

    tuning_results.append({
        "Model": name,
        "Best Parameters": str(grid.best_params_),
        "Best CV F1": grid.best_score_
    })

    # Save trained model
    joblib.dump(
        grid.best_estimator_,
        f"models/{filename}.pkl"
    )

    # Select best model
    if grid.best_score_ > best_score:
        best_score = grid.best_score_
        best_model = grid.best_estimator_
        best_name = name

# 8. Save results
results_df = pd.DataFrame(results)
cv_df = pd.DataFrame(cv_results)
tuning_df = pd.DataFrame(tuning_results)

results_df.to_csv(
    "results/model_comparison.csv", index=False
)

cv_df.to_csv(
    "results/cross_validation.csv", index=False
)

tuning_df.to_csv(
    "results/tuning_results.csv", index=False
)

# 9. Model comparison graph
results_df.set_index("Model").plot(
    kind="bar", figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=150)
plt.close()

# 10. Cross-validation graph
plt.figure(figsize=(8, 5))

sns.barplot(
    data=cv_df, x="Model", y="Mean F1"
)

plt.title("5-Fold Cross-Validation")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.savefig("graphs/cross_validation.png", dpi=150)
plt.close()

# 11. Final testing
predictions = best_model.predict(X_test)

print("\n===== BEST MODEL =====")
print("Best model:", best_name)
print("Test accuracy:", accuracy_score(y_test, predictions))

report = classification_report(
    y_test, predictions,
    labels=[0, 1],
    target_names=["Normal", "Malicious"],
    zero_division=0
)

print(report)

with open("results/final_report.txt", "w") as file:
    file.write("Best Model: " + best_name + "\n\n")
    file.write(report)

# Final confusion matrix
cm = confusion_matrix(
    y_test, predictions, labels=[0, 1]
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm, annot=True, fmt="d", cmap="Greens",
    xticklabels=["Normal", "Malicious"],
    yticklabels=["Normal", "Malicious"]
)

plt.title("Best Model - Test Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("graphs/best_model_confusion_matrix.png",
            dpi=150)
plt.close()

print("False negatives:", cm[1, 0])

# 12. ROC curve
probabilities = best_model.predict_proba(X_test)[:, 1]

fpr, tpr, thresholds = roc_curve(
    y_test, probabilities
)

roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Best Model")
plt.legend()
plt.tight_layout()
plt.savefig("graphs/roc_curve.png", dpi=150)
plt.close()

# 13. Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save unseen test records for dashboard
test_data = X_test.copy()
test_data["label"] = y_test

test_data.to_csv(
    "results/test_data.csv", index=False
)

# Final results
print("\n===== MODEL COMPARISON =====")
print(results_df.to_string(index=False))

print("\nBest model:", best_name)
print("ROC AUC:", roc_auc)
print("All models saved successfully!")
print("MODEL TRAINING COMPLETED!")
