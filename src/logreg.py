# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.metrics import accuracy_score


# Load and preprocess data here
def define_preprocessor(X_train):
    """
    Defines a preprocessing pipeline for numeric and categorical features.

    Parameters:
    - X_train: The training data containing features.

    Returns:
    - preprocessor (ColumnTransformer): A column transformer object that applies the specified transformations to numeric and categorical columns.
    """
    numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X_train.select_dtypes(include=["object"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def feature_selection_and_importance(X_train_scaled, y_train, X_train_columns):
    """
    Applies various feature selection techniques and plots the importance of selected features.

    Parameters:
    - X_train_scaled: Scaled training data.
    - y_train: Target variable for the training data.
    - X_train_columns: Column names of the training data before scaling.

    Outputs:
    - Visualizations of feature importances and prints selected features using different selection methods.
    """
    # Random Forest for feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)

    # Plotting feature importance for Random Forest
    plt.figure(figsize=(10, 6))
    importance_rf = pd.Series(rf.feature_importances_, index=X_train_columns)
    importance_rf.nlargest(10).plot(kind="barh")
    plt.title("Top 10 Features Importance - Random Forest")
    plt.show()

    # Univariate Feature Selection
    selector = SelectKBest(score_func=f_classif, k=10)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    selected_features_indices = selector.get_support(indices=True)
    selected_features = X_train_columns[selected_features_indices]

    # Lasso Regularization
    lasso = LassoCV(cv=5, random_state=42)
    lasso.fit(X_train_scaled, y_train)

    # Plotting feature importance for Lasso
    plt.figure(figsize=(10, 6))
    importance_lasso = pd.Series(np.abs(lasso.coef_), index=X_train_columns)
    importance_lasso.nlargest(10).plot(kind="barh")
    plt.title("Top 10 Features Importance - Lasso Regularization")
    plt.show()

    # Recursive Feature Elimination (RFE) with Logistic Regression
    logreg = LogisticRegression(max_iter=10000)
    rfe = RFE(estimator=logreg, n_features_to_select=10, step=1)
    rfe.fit(X_train_scaled, y_train)

    # Feature ranking from RFE
    rfe_ranking = pd.Series(rfe.ranking_, index=X_train_columns)
    selected_features_rfe = rfe_ranking[rfe_ranking == 1].index

    # Output selected features from different techniques
    print("Selected features using Random Forest Feature Importance:")
    print(importance_rf.nlargest(10))
    print("\nSelected features using Univariate Feature Selection:")
    print(selected_features)
    print("\nSelected features using Lasso Regularization:")
    print(importance_lasso.nlargest(10))
    print("\nSelected features using Recursive Feature Elimination (RFE):")
    print(selected_features_rfe)


def main():
    # Load and preprocess the data
    county_aggregation = pd.read_csv("../data/countyagg.csv")
    X = county_aggregation.drop(
        [
            "State",
            "County",
            "is_food_desert",
            "LowIncomeTracts",
            "LILATracts_1And10",
            "LALOWI1_10",
        ],
        axis=1,
    )
    y = county_aggregation["is_food_desert"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = define_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)

    # Feature selection and importance
    feature_selection_and_importance(X_train_scaled, y_train, X_train.columns)


if __name__ == "__main__":
    main()
