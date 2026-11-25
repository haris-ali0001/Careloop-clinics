import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/appointments_clean.csv")
df['no_show'] = (df['status'] == 'No-Show').astype(int)

# reminder Yes/No -> 1/0
df['reminder_sent'] = df['reminder_sent'].map({'Yes': 1, 'No': 0})

# 8 features - patient_id, appointment_id, dates, clinic_id, gender, status NOT used
features = ['age','distance_km','fee_inr','lead_days','prior_no_shows','reminder_sent','booking_channel','visit_type']
X = pd.get_dummies(df[features], columns=['booking_channel','visit_type']).astype(float)

y = df['no_show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X columns:", X.shape[1])
print(list(X.columns))
print(f"Train: {len(X_train)}, Test: {len(X_test)}")
print(f"Test no-show: {y_test.sum()}")
print(f"Baseline: {(y_test==0).mean():.4f}")
# ---------- 3.2 Logistic ----------
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

lr_pipe = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=1000))])
cv_auc = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="roc_auc").mean()
oof = cross_val_predict(lr_pipe, X_train, y_train, cv=5, method="predict_proba")[:, 1]
lr_pipe.fit(X_train, y_train)
probs = lr_pipe.predict_proba(X_test)[:, 1]

print(f"\n--- 3.2 Logistic ---")
print(f"CV ROC-AUC: {cv_auc:.3f}")
print(f"Test ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
print(f"Accuracy @0.5: {accuracy_score(y_test, (probs>=0.5).astype(int)):.3f}")
print("Confusion @0.5:\n", confusion_matrix(y_test, (probs>=0.5).astype(int)))
print(classification_report(y_test, (probs>=0.5).astype(int)))

print("Coefficients:")
for n,c in sorted(zip(X.columns, lr_pipe.named_steps['model'].coef_[0]), key=lambda x: x[1]):
    print(f"{n}: {c:.3f}")
    # ---------- 3.3 Decision Tree ----------
from sklearn.tree import DecisionTreeClassifier

print("\n--- 3.3 Decision Tree ---")

for name, tree in [("Full-depth", DecisionTreeClassifier(random_state=42)),
                   ("Depth=3", DecisionTreeClassifier(max_depth=3, random_state=42))]:
    tree.fit(X_train, y_train)
    pred = tree.predict(X_test)
    prob = tree.predict_proba(X_test)[:,1]
    print(f"{name} - Depth: {tree.get_depth()}, Train: {tree.score(X_train, y_train):.3f}, Test: {accuracy_score(y_test, pred):.3f}, ROC-AUC: {roc_auc_score(y_test, prob):.3f}")
    # ---------- 3.4 Random Forest ----------
from sklearn.ensemble import RandomForestClassifier

print("\n--- 3.4 Random Forest ---")
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
pred = rf.predict(X_test)
prob = rf.predict_proba(X_test)[:,1]
print(f"RF - Train: {rf.score(X_train, y_train):.3f}, Test: {accuracy_score(y_test, pred):.3f}, ROC-AUC: {roc_auc_score(y_test, prob):.3f}")
print("Importances:")
for n,i in sorted(zip(X.columns, rf.feature_importances_), key=lambda x: -x[1]):
    print(f"{n}: {i:.3f}")
    # ---------- 3.5 Tuning + Winner ----------
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

print("\n--- 3.5 Tuning (CV ROC-AUC) ---")

# Logistic
lr_grid = GridSearchCV(lr_pipe, {"model__C": [0.1, 1, 10]}, cv=5, scoring="roc_auc")
lr_grid.fit(X_train, y_train)
print(f"Logistic best_score: {lr_grid.best_score_:.3f}, best_params: {lr_grid.best_params_}")

# Tree
tree_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
    {"max_depth": [3, 5, None], "min_samples_leaf": [1, 5]}, cv=5, scoring="roc_auc")
tree_grid.fit(X_train, y_train)
print(f"Tree best_score: {tree_grid.best_score_:.3f}, best_params: {tree_grid.best_params_}")

# Random Forest
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42, n_jobs=-1),
    {"n_estimators": [300], "max_depth": [None, 10], "min_samples_leaf": [1, 5]}, cv=5, scoring="roc_auc")
rf_grid.fit(X_train, y_train)
print(f"RF best_score: {rf_grid.best_score_:.3f}, best_params: {rf_grid.best_params_}")
print(f"Tuned RF Test ROC-AUC: {roc_auc_score(y_test, rf_grid.predict_proba(X_test)[:,1]):.3f}")

# Winner = Logistic at threshold 0.35
best_lr = lr_grid.best_estimator_
probs = best_lr.predict_proba(X_test)[:,1]
pred35 = (probs >= 0.35).astype(int)
print(f"\nWINNER: Logistic @0.35 - Flagged: {pred35.sum()}, Precision: {precision_score(y_test, pred35):.3f}, Recall: {recall_score(y_test, pred35):.3f}, F1: {f1_score(y_test, pred35):.3f}")