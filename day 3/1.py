# Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier # Добавили GB
from sklearn.model_selection import cross_val_score # Добавили функцию оценки
from sklearn.model_selection import GridSearchCV
df = pd.read_csv('german_data_creditcard(in).csv')
X = df.drop("Creditability", axis=1)
y = df["Creditability"]

# print("Shape of X:", X.shape)
# print("Shape of y:", y.shape)
# print(X.dtypes)

# need to understand which columns are numeric and which are categorical, because it understands only number
categorical = X.select_dtypes(include="object").columns
numeric = X.select_dtypes(include=["int64", "float64"]).columns
# to ensure better learning and no data skipping, we need categorical data encode


preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric)
    ]
)

rf = RandomForestClassifier(random_state=42)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

#define search parameters to find optinal balance between model complexity and accuracy
params = {
    'classifier__max_depth': [2, 3, 5, 10],
    'classifier__min_samples_leaf': [1,3,5,7,9,15],
    'classifier__n_estimators': [10, 25, 30, 50, 100, 200]
}


grid_search = GridSearchCV(pipeline, params, cv=5, scoring='accuracy')
grid_search.fit(X, y)


print("best parameters:", grid_search.best_params_)
print("best score:", grid_search.best_score_)

# task 2
# create models
ada_model = AdaBoostClassifier(n_estimators=100, random_state=42)
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# create pipelines
pipeline_ada = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", ada_model)])
pipeline_gb = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", gb_model)])

# checking score without gridsearch, average accuracy 
score_ada = cross_val_score(pipeline_ada, X, y, cv=5, scoring='accuracy').mean()
score_gb = cross_val_score(pipeline_gb, X, y, cv=5, scoring='accuracy').mean()

print("adaboost accuracy: ",score_ada)
print("gradient boosting accuracy: ", score_gb)