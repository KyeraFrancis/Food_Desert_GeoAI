# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    ConfusionMatrixDisplay,
)

from sklearn.tree import plot_tree


# Function to load and preprocess the data
def load_and_preprocess_data():
    # Load the data
    county_aggregation = pd.read_csv("../data/countyagg.csv")

    county_aggregation["is_food_desert"] = (
        (county_aggregation["LowIncomeTracts"] > 0)
        & (county_aggregation["LILATracts_1And10"] > 0)
    ).astype(int)

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

    return X, y


# Function to split the dataset
def split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)


# Function to define the model pipeline
def define_model_pipeline(X):
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = X.select_dtypes(include=["object"]).columns

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

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )

    return model


# Function to fit the model and evaluate
def fit_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix for Random Forest Classifier")
    ax.grid(False)

    print(classification_report(y_test, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Roc Curve for Random Forest Classifier")
    plt.legend(loc="lower right")
    plt.show()


# Main function to orchestrate the modeling process
def main():
    X, y = load_and_preprocess_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = define_model_pipeline(X)
    fit_and_evaluate(model, X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
