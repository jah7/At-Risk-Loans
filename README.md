# Understanding At-Risk Loans

This project contains a Jupyter Notebook that trains and evaluates classification models to identify at-risk loans. It leverages **Logistic Regression** and **XGBoost** classifiers using **scikit-learn** and **XGBoost**, producing standard performance metrics and visualizations such as ROC and precision-recall curves.

> **Note:** The notebook is code-focused and contains minimal explanatory markdown. This README provides instructions for running the notebook, expected inputs, and guidance for customization.

## Quick Summary

The notebook performs the following steps:

1. **Data Loading**

   * Expects input as a CSV file or a pandas DataFrame containing loan features and the target variable (`bad_flag` or similar).

2. **Data Preprocessing**

   * Performs a train/test split.
   * Applies standard scaling to features.

3. **Model Training**

   * Trains at least two models: `LogisticRegression` and `XGBClassifier`.

4. **Evaluation**

   * Computes performance metrics: classification report, ROC AUC, precision-recall AUC.
   * Plots diagnostics: ROC and precision-recall curves (and potentially other charts).
   * Prints evaluation outputs and displays plots inline.

## Results

* Precision-recall curves show steep early drops, indicating that only a small fraction of positive cases can be captured before precision declines.
* This behavior is common in **imbalanced binary classification problems**, where the model favors the majority class.
* **XGBoost** slightly outperforms Logistic Regression on training data but shows similar performance on unseen test data.

### Interpretation

* Models are good at predicting the majority class but fail to capture enough minority (at-risk) cases.
* Improving recall for the minority class should be a priority.

## Next Steps / Recommendations

1. **Address Class Imbalance**

   * Use class weighting (`class_weight="balanced"` for Logistic Regression).
   * Consider sampling techniques such as SMOTE, oversampling the minority class, or undersampling the majority class.

2. **Hyperparameter Tuning**

   * Adjust XGBoost parameters (`max_depth`, `learning_rate`, `scale_pos_weight`) to improve recall without sacrificing precision.

3. **Feature Engineering**

   * Introduce new or transformed features to improve separability between classes.

4. **Model Evaluation**

   * Focus on ROC AUC or PR AUC for performance tracking, as accuracy can be misleading for imbalanced datasets.

## Usage

1. Ensure dependencies are installed:

```bash
pip install pandas scikit-learn xgboost matplotlib seaborn
```

2. Load your dataset into a pandas DataFrame or provide a CSV file path in the notebook.

3. Run the notebook cells sequentially to preprocess data, train models, and view evaluation metrics and plots.
