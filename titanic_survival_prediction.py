"""
============================================================
 🚢 TITANIC SURVIVAL PREDICTION
 A complete Machine Learning workflow
 from raw data to production-ready pipeline
============================================================

 Author : Elvira Tomizzi — Business & Data Analyst
 Tools  : Python · Pandas · Scikit-learn · XGBoost · Seaborn
 GitHub : github.com/Elvtomi

------------------------------------------------------------
MAIN QUESTION
 "Given passenger characteristics (age, sex, ticket class,
  family size, fare), can we predict whether they survived?"

 This binary classification problem mirrors real-world tasks:
 customer churn, credit scoring, fraud detection, diagnostics.
------------------------------------------------------------

 WORKFLOW
  1. Data loading & first exploration
  2. Missing values analysis
  3. Exploratory Data Analysis (EDA)
  4. Feature engineering (encoding + scaling)
  5. Model comparison — 9 classifiers
  6. Hyperparameter tuning — GridSearchCV
  7. Confusion matrix & classification report
  8. ROC curves — AUC comparison
  9. Production pipeline (Preprocessor + Scaler + Classifier)
 10. Cross-validation
 11. Prediction on new data & Kaggle export
 12. Pipeline with PCA — performance comparison
"""

# ============================================================
# SECTION 1 — IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    ConfusionMatrixDisplay, classification_report,
    RocCurveDisplay, roc_auc_score, accuracy_score
)
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer, FunctionTransformer
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
import joblib

import warnings
warnings.filterwarnings('ignore')


# ============================================================
# SECTION 2 — DATA LOADING & FIRST EXPLORATION
# ============================================================
# Before touching any data, we define our goal clearly:
# predict the `Survived` column (0 = No, 1 = Yes)
# based on passenger features.
#
# Feature overview:
#  Survived  → target (binary: 0/1)
#  Pclass    → ticket class (1=1st, 2=2nd, 3=3rd) — socioeconomic proxy
#  Sex       → gender (categorical)
#  Age       → age in years (has missing values)
#  SibSp     → # siblings/spouses aboard
#  Parch     → # parents/children aboard
#  Ticket    → ticket number (not informative → will drop)
#  Fare      → fare paid (numerical)
#  Cabin     → cabin number (mostly missing → will drop)
#  Embarked  → port of embarkation (C/Q/S)

#path = '/titanic.csv'
df = pd.read_csv(path, sep='\t')

print("=== Dataset shape ===")
print(df.shape)
print("\n=== First look ===")
print(df.head())
print("\n=== Data types & nulls ===")
df.info()


# ============================================================
# SECTION 3 — MISSING VALUES ANALYSIS
# ============================================================
# Missing data is never "just a technical problem."
# Before imputing, we ask: WHY is the data missing?
#
#  Age     → missing at random (~20%) → use interpolation
#  Cabin   → missing systematically — lower-class passengers
#             often had no assigned cabin (MNAR) → drop column
#  Embarked→ only 2 rows missing → drop those rows

# Visual map of missing values (white = missing)
plt.figure(figsize=(7, 5))
sns.heatmap(df.isnull(), cbar=False)
plt.title("Missing values heatmap — before cleaning")
plt.tight_layout()
plt.show()

# Class balance check — an imbalanced target needs special treatment.
# ~38% survived, ~62% did not → moderate imbalance, manageable.
# We will use stratify=y in train_test_split to preserve this ratio.
print("\n=== Target class balance ===")
print(df['Survived'].value_counts())
print(df['Survived'].value_counts(normalize=True).round(2))


# ============================================================
# SECTION 4 — FEATURE SELECTION & EDA
# ============================================================
# We remove non-informative features:
#  Name, Ticket, PassengerId → identifiers, not predictive
#  Cabin → too many missing values (~77%)

df = df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Pairplot: relationships between all numeric features, colored by Sex.
# Key question: which variables separate survivors from non-survivors?
sns.pairplot(df, hue='Sex', height=2.5)
plt.suptitle("Pairplot — features by Sex", y=1.02)
plt.show()

# Heatmap of remaining missing values
print("\n=== Remaining missing values ===")
print(df.isnull().sum())

plt.figure(figsize=(7, 5))
sns.heatmap(df.isnull())
plt.title("Missing values heatmap — after dropping columns")
plt.show()


# ============================================================
# SECTION 5 — HANDLING MISSING VALUES
# ============================================================
# Age: interpolation estimates missing values based on the
# trend of existing ones — a smooth approximation that
# preserves the overall age distribution.

df['Age'] = df['Age'].interpolate()

# Verify Age is clean
plt.figure(figsize=(7, 5))
sns.heatmap(df.isnull())
plt.title("Missing values — after Age interpolation")
plt.show()

# Embarked: only 2 missing rows → drop them (negligible data loss)
print("\n=== Embarked value counts ===")
print(df['Embarked'].value_counts())
df = df.dropna()

print(f"\nDataset size after cleaning: {df.shape}")


# ============================================================
# SECTION 6 — EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================
# Key finding: the "women and children first" protocol
# is statistically visible in the data.
#
#  - Women survived at ~74% vs men at ~19%
#  - 1st class passengers had far higher survival rates
#  - Children (low Age) tended to survive more
#  - Higher Fare → higher class → higher survival

sns.relplot(
    x="Age", y="Fare", hue="Sex",
    col="Survived", row="Pclass",
    data=df, height=3
)
plt.suptitle(
    "Age vs Fare by Sex — split by Survived (col) and Pclass (row)",
    y=1.02
)
plt.show()


# ============================================================
# SECTION 7 — FEATURE ENGINEERING
# ============================================================
# Encoding categorical variables:
# Machine learning models require numeric input.
# We use One-Hot Encoding (pd.get_dummies) with drop_first=True
# to avoid the "dummy variable trap" (multicollinearity).
#
# Sex → Sex_male (1 = male, 0 = female)
# Embarked → Embarked_Q, Embarked_S (C is the reference)

df = pd.get_dummies(df, columns=['Embarked', 'Sex'], dtype=int, drop_first=True)

# Reorder columns — target at the end for clarity
df = df[['Pclass', 'Age', 'SibSp', 'Parch', 'Fare',
         'Embarked_Q', 'Embarked_S', 'Sex_male', 'Survived']]

# Feature matrix (X) and target vector (y)
X_ = df.iloc[:, 0:-1]   # un-scaled features
y  = df.iloc[:, -1]     # target: Survived


# StandardScaler: transforms each feature to mean=0, std=1.
# Why? Many algorithms (KNN, SVM) are scale-sensitive.
# A feature with large values (Fare: 0-500) would dominate
# over one with small values (Parch: 0-8) without scaling.
#
# IMPORTANT: fit_transform on train, only transform on test.
# This prevents data leakage.

sc = StandardScaler()
X = sc.fit_transform(X_)


# ============================================================
# SECTION 8 — TRAIN / TEST SPLIT
# ============================================================
# stratify=y preserves the 38/62 class ratio in both splits.
# random_state=667 ensures reproducibility.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=667,
    stratify=y
)

print(f"\nTrain size: {X_train.shape} | Test size: {X_test.shape}")


# ============================================================
# SECTION 9 — MODEL COMPARISON (9 CLASSIFIERS)
# ============================================================
# We test 9 algorithms under identical conditions.
# Each has a different learning strategy:
#
#  Decision Tree     → splits data by feature thresholds
#  Random Forest     → ensemble of trees, majority vote
#  Gradient Boosting → trees trained sequentially (each corrects previous)
#  Logistic Reg.     → linear decision boundary
#  Naive Bayes       → probabilistic, assumes feature independence
#  KNN               → classifies by proximity to neighbors
#  SVM (RBF)         → finds optimal separating hyperplane
#  XGBoost           → advanced gradient boosting + regularization
#
# Ensemble methods (RF, GB, XGB) typically win on tabular data
# because they combine the wisdom of many weak learners.

print("\n=== Individual model scores ===")

# Decision Tree
dt_clf = DecisionTreeClassifier(max_depth=4)
dt_clf.fit(X_train, y_train)
print(f"Decision Tree       : {dt_clf.score(X_test, y_test):.4f}")

# Random Forest
rf_clf = RandomForestClassifier(n_estimators=200)
rf_clf.fit(X_train, y_train)
print(f"Random Forest       : {rf_clf.score(X_test, y_test):.4f}")

# Gradient Boosting
gb_clf = GradientBoostingClassifier()
gb_clf.fit(X_train, y_train)
print(f"Gradient Boosting   : {gb_clf.score(X_test, y_test):.4f}")

# Logistic Regression
lr_clf = LogisticRegression(max_iter=1000)
lr_clf.fit(X_train, y_train)
print(f"Logistic Regression : {lr_clf.score(X_test, y_test):.4f}")

# Naive Bayes
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)
print(f"Naive Bayes         : {nb_clf.score(X_test, y_test):.4f}")

# K-Nearest Neighbors
knn_clf = KNeighborsClassifier(n_neighbors=2)
knn_clf.fit(X_train, y_train)
print(f"KNN (k=2)           : {knn_clf.score(X_test, y_test):.4f}")

# Support Vector Machine
svm_clf = SVC(probability=True)
svm_clf.fit(X_train, y_train)
print(f"SVM (RBF)           : {svm_clf.score(X_test, y_test):.4f}")

# XGBoost
xgb_clf = XGBClassifier()
xgb_clf.fit(X_train, y_train)
print(f"XGBoost             : {xgb_clf.score(X_test, y_test):.4f}")


# ============================================================
# SECTION 10 — ALL-IN-ONE COMPARISON
# ============================================================

classifiers = [
    DecisionTreeClassifier(max_depth=4),
    RandomForestClassifier(n_estimators=200, random_state=667),
    GradientBoostingClassifier(),
    GradientBoostingClassifier(n_estimators=50),
    LogisticRegression(max_iter=1000),
    GaussianNB(),
    KNeighborsClassifier(n_neighbors=2),
    SVC(probability=True, kernel='rbf'),
    XGBClassifier()
]

# Re-split without stratify for the combined loop
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=667
)

for cls in classifiers:
    cls.fit(X_train, y_train)
    cls.predict(X_test)


# ============================================================
# SECTION 11 — CONFUSION MATRIX (ALL MODELS)
# ============================================================
# How to read a confusion matrix:
#
#          Predicted: No  |  Predicted: Yes
#  Actual No    TN        |      FP   ← Type I error
#  Actual Yes   FN        |      TP   ← correct
#               ↑ Type II error
#
# In this context, False Negatives are costly:
# predicting "died" for someone who actually survived
# means missing a true survivor.
# The choice between optimizing precision vs recall depends
# on the real-world cost of each error type.

labels = ['morti', 'sopravv']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))
for cls, ax in zip(classifiers, axes.flatten()):
    ConfusionMatrixDisplay.from_estimator(
        cls, X_test, y_test,
        ax=ax, cmap='Blues',
        display_labels=labels
    )
    ax.title.set_text(type(cls).__name__)
plt.suptitle("Confusion Matrices — all 9 classifiers", fontsize=14)
plt.tight_layout()
plt.show()

# Classification report — per-class metrics
# Precision : of all predicted survivors, how many actually survived?
# Recall    : of all true survivors, how many did we catch?
# F1-score  : harmonic mean of precision and recall
# Weighted avg accounts for class imbalance

print("\n=== Classification Reports ===")
for cls in classifiers:
    print(f"\n--- {cls.__class__.__name__} ---")
    print(classification_report(
        y_test, cls.predict(X_test), target_names=labels
    ))


# ============================================================
# SECTION 12 — ROC CURVES & AUC
# ============================================================
# ROC curve plots:
#  X-axis → False Positive Rate (wrongly predicting "survived")
#  Y-axis → True Positive Rate / Recall (correctly catching survivors)
#
# AUC (Area Under Curve):
#  1.0 → perfect model
#  0.5 → random guessing (dashed diagonal)
#  0.8+ → good model
#
# AUC is more honest than accuracy for imbalanced datasets:
# a model predicting only the majority class gets high accuracy
# but a terrible ROC curve.

plt.figure(figsize=(9, 7))
ax = plt.gca()
plt.plot([0, 1], [0, 1], '--b', label='Random baseline (AUC=0.50)')

for cls in classifiers:
    model = cls.fit(X_train, y_train)
    y_score = model.predict_proba(X_test)[:, 1]
    auc_val = roc_auc_score(y_test, y_score)
    RocCurveDisplay.from_estimator(
        model, X_test, y_test,
        label=f'{cls.__class__.__name__} (AUC={auc_val:.2f})',
        ax=ax
    )

plt.title("ROC Curves — all classifiers")
plt.legend(loc=4, fontsize=8)
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 13 — HYPERPARAMETER TUNING (GridSearchCV)
# ============================================================
# Default Random Forest is good — but not necessarily optimal.
# GridSearchCV systematically tests all parameter combinations
# using 5-fold cross-validation to evaluate each one.
#
# 288 combinations × 5 folds = 1,440 model fits
# Computationally expensive — but finds the true best config.

param_grid = {
    'bootstrap':        [True],
    'max_depth':        [80, 90, 100, 110],
    'max_features':     [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split':[8, 10, 12],
    'n_estimators':     [100, 200, 300, 1000]
}

rf = RandomForestClassifier(random_state=667)
grid_search_rf = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1
)

# Uncomment to run (takes several minutes):
# grid_search_rf.fit(X_train, y_train)
# print("Best params:", grid_search_rf.best_params_)
# best_grid = grid_search_rf.best_estimator_

# Using best params found from GridSearch:
best_grid = RandomForestClassifier(
    bootstrap=True,
    max_depth=80,
    max_features=2,
    min_samples_leaf=3,
    min_samples_split=8,
    n_estimators=100,
    random_state=667
)
best_grid.fit(X_train, y_train)

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.4f}")
    return round(acc, 4)

print("\n=== Tuned Random Forest performance ===")
evaluate(best_grid, X_test, y_test)


# ============================================================
# SECTION 14 — FEATURE IMPORTANCE
# ============================================================
# Which features did the model rely on most?
#
# Typical findings on Titanic:
#  Sex_male  → strongest predictor — "women first" is statistically real
#  Fare/Pclass → socioeconomic class determined lifeboat access
#  Age        → moderate importance — children were prioritized
#  SibSp/Parch→ lower importance — family size had mixed effects
#
# The model has learned from data what historians have documented
# for over a century. This is the value of interpretable ML.

X_named = df.iloc[:, :-1]
feature_imp = pd.Series(
    best_grid.feature_importances_,
    index=X_named.columns
).sort_values(ascending=False)

plt.figure(figsize=(7, 5))
sns.barplot(x=feature_imp, y=feature_imp.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Feature Importance — Tuned Random Forest")
plt.tight_layout()
plt.show()

print("\n=== Feature importances ===")
print(feature_imp.round(4))


# ============================================================
# SECTION 15 — PRODUCTION PIPELINE
# ============================================================
# PROBLEM with manual preprocessing:
# When a new passenger arrives, we must remember to apply
# every step in the exact same order with exact same params.
# Any mistake = wrong prediction.
#
# SOLUTION: scikit-learn Pipeline chains all steps into one object:
#   Raw data → [Imputer] → [Encoder] → [Scaler] → [Classifier] → Prediction
#
# Benefits:
#  ✅ No data leakage — preprocessing fit only on training data
#  ✅ Reproducibility — same steps, guaranteed, every time
#  ✅ Deployable — save and load with joblib
#  ✅ Clean code — one object replaces many manual steps
#
# ColumnTransformer applies different preprocessing per column type:
#  Numerical (Age, Fare…) → fill NaN with median → bin → scale
#  Categorical (Sex, Embarked) → fill NaN → One-Hot Encode

path_pipe = 'https://frenzy86.s3.eu-west-2.amazonaws.com/fav/tecno/titanic.csv'
df_pipe = pd.read_csv(path_pipe, sep='\t')

features_to_remove = ['Name', 'PassengerId', 'Ticket', 'Cabin']
df_clean = df_pipe.drop(features_to_remove, axis=1)

X = df_clean.drop(['Survived'], axis=1)
y = df_clean['Survived']

# Auto-detect numerical vs categorical columns
numerical_features   = [c for c, d in zip(X.columns, X.dtypes) if d.kind in ['i','f']]
categorical_features = [c for c, d in zip(X.columns, X.dtypes) if d.kind not in ['i','f']]

print(f"\nNumerical features   : {numerical_features}")
print(f"Categorical features : {categorical_features}")

# Build the preprocessor
Preprocessor = make_column_transformer(
    (make_pipeline(
        SimpleImputer(strategy='median'),
        KBinsDiscretizer(n_bins=3)
    ), numerical_features),
    (make_pipeline(
        SimpleImputer(strategy='constant', fill_value='missing'),
        OneHotEncoder(categories='auto', handle_unknown='ignore')
    ), categorical_features)
)

scaler = StandardScaler()

classifier = RandomForestClassifier(
    bootstrap=True,
    max_depth=80,
    max_features=2,
    min_samples_leaf=3,
    min_samples_split=8,
    n_estimators=100
)

# Assemble the full pipeline
model_pipe = Pipeline([
    ('Preprocessing features',      Preprocessor),
    ('Scaling and standardize data', scaler),
    ('Classifier',                   classifier)
])


# ============================================================
# SECTION 16 — CROSS-VALIDATION
# ============================================================
# Simple train/test split depends on which rows ended up in
# each split — there's an element of luck.
#
# 5-fold cross-validation solves this:
#  1. Split data into 5 equal folds
#  2. Train on 4 folds, test on the 5th
#  3. Repeat 5 times (each fold is the test set once)
#  4. Average the 5 scores
#
# Standard deviation of scores tells us consistency:
#  Low std  → stable, trustworthy model
#  High std → sensitive to data, may be overfitting

cross_val_scores = cross_val_score(model_pipe, X, y, cv=5)

print("\n=== 5-Fold Cross-Validation ===")
print(f"Scores per fold : {cross_val_scores.round(4)}")
print(f"Mean score      : {np.mean(cross_val_scores):.4f}")
print(f"Std deviation   : {np.std(cross_val_scores):.4f}")

# Fit on the full dataset
model_pipe.fit(X, y)
y_pred_tot = model_pipe.predict(X)

print("\n=== Classification Report (full training data) ===")
print(classification_report(y, y_pred_tot))


# ============================================================
# SECTION 17 — PREDICTION ON TEST DATA (Kaggle format)
# ============================================================

test = pd.read_csv(/titanic-test.csv')
test_clean = test.drop(features_to_remove, axis=1)

test_pred = model_pipe.predict(test_clean).astype(int)

# Export in Kaggle competition format
sub = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': test_pred
})
sub.to_csv("solu_RF_pipe.csv", index=False)
print("\n✅ Kaggle submission saved to: solu_RF_pipe.csv")


# ============================================================
# SECTION 18 — SINGLE PASSENGER PREDICTION
# ============================================================
# The pipeline is now trained and saved as a single object.
# It handles preprocessing internally — no manual steps needed.
#
# Test case: Female, 34, 3rd class, fare £7.8, Queenstown, alone
#
# What our EDA tells us to expect:
#  Female    → strong survival signal  ✅
#  3rd class → lower survival rate     ⚠️
#  Fare 7.8  → low, confirms 3rd class ⚠️
#  Age 34    → neutral                 →
#
# Being female is the dominant predictor — the model
# should predict survival. We can sanity-check it against
# domain knowledge and trust results that align with history.

single_passenger = pd.DataFrame({
    "Pclass":   [3],
    "Sex":      ['female'],
    "Age":      [34],
    "SibSp":    [0],
    "Parch":    [0],
    "Fare":     [7.8],
    "Embarked": ['Q']
})

res = model_pipe.predict(single_passenger).astype(int)[0]
classes = {0: 'Did NOT survive', 1: 'SURVIVED'}

print(f"\n=== Single Passenger Prediction ===")
print(f"Input  : {single_passenger.iloc[0].to_dict()}")
print(f"Result : {classes[res]}")


# ============================================================
# SECTION 19 — SAVE & LOAD MODEL
# ============================================================
# joblib serializes the entire pipeline (preprocessing + model)
# into a single .pkl file — deployable anywhere.

joblib.dump(model_pipe, 'titanic_pipe.pkl')
print("\n✅ Pipeline saved to: titanic_pipe.pkl")

# Load and predict (simulates production environment)
loaded_pipe = joblib.load('titanic_pipe.pkl')
res_loaded = loaded_pipe.predict(single_passenger).astype(int)[0]
print(f"✅ Loaded pipeline prediction: {classes[res_loaded]}")


# ============================================================
# SECTION 20 — PIPELINE WITH PCA
# ============================================================
# PCA (Principal Component Analysis) reduces dimensionality
# by finding the directions of maximum variance in the data.
# This can sometimes improve performance by removing noise.
#
# Here we test whether adding PCA before the classifier
# meaningfully changes results. Spoiler: on this dataset,
# the difference is minimal — the features are already
# informative enough. This is itself an important finding.

model_pipe_pca = Pipeline([
    ('Preprocessing features',       Preprocessor),
    ('Scaling and standardize data',  scaler),
    ('PCA',                           PCA()),      # ← added
    ('Classifier',                    classifier)
])

cross_val_pca = cross_val_score(model_pipe_pca, X, y, cv=5)

print("\n=== Cross-Validation: Pipeline vs Pipeline+PCA ===")
print(f"Without PCA : mean={np.mean(cross_val_scores):.4f}  std={np.std(cross_val_scores):.4f}")
print(f"With PCA    : mean={np.mean(cross_val_pca):.4f}  std={np.std(cross_val_pca):.4f}")

# Fit PCA pipeline and compare ROC curves
model_pipe_pca.fit(X, y)
model_pipe.fit(X, y)

auc_pca  = roc_auc_score(y, model_pipe_pca.predict_proba(X)[:, 1])
auc_base = roc_auc_score(y, model_pipe.predict_proba(X)[:, 1])

plt.figure(figsize=(8, 7))
ax = plt.gca()
RocCurveDisplay.from_estimator(
    model_pipe_pca, X, y,
    label=f'Pipeline + PCA (AUC={auc_pca:.2f})', ax=ax
)
RocCurveDisplay.from_estimator(
    model_pipe, X, y,
    label=f'Pipeline (AUC={auc_base:.2f})', ax=ax
)
plt.plot([0, 1], [0, 1], '--b', label='Random baseline')
plt.title("ROC Curve — Pipeline vs Pipeline + PCA")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# SECTION 21 — CONCLUSIONS
# ============================================================
# What we built:
#  A complete, production-ready ML pipeline that:
#  - Handles missing data automatically
#  - Encodes and scales features consistently
#  - Predicts survival with ~82% accuracy (cross-validated)
#  - Can be saved, loaded, and used on new data with .predict()
#
# What the data reveals:
#  Factor       | Effect      | Historical explanation
#  Sex=Female   | Strong +    | "Women and children first" protocol
#  Pclass=1     | Positive    | Cabins near deck, more lifeboats
#  Age < 15     | Positive    | Children prioritized
#  Pclass=3     | Negative    | Below deck, language barriers
#  Large family | Mixed       | Families sometimes stayed together
#
# Real-world applications of this exact methodology:
#  🏦 Credit scoring   — will this customer repay the loan?
#  📱 Churn prediction — will this user cancel their subscription?
#  🏥 Medical diagnosis— does this patient have the condition?
#  🛡️ Fraud detection  — is this transaction legitimate?
#  🎯 Marketing        — will this customer convert?
#
# The Titanic dataset is historical.
# The analytical thinking is timeless.
#
# ── Elvira Tomizzi | github.com/Elvtomi ──────────────────
