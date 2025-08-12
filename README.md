# Decision-trees-Vs-random-Forests
This project compares the performance of Decision Tree and Random Forest classifiers on a heart disease dataset. It includes data cleaning, model training, visualization of the decision tree, feature importance analysis, and cross-validation for reliable performance evaluation.
Heart Disease Prediction – Decision Tree & Random Forest

Required libraries
--------------------------
1.Scikit-learn

2.Graphviz

-------------------
Installation

pip install scikit-learn graphviz

-----------------------------

Data Cleaning Summary:

It is clean data set only duplicates where removed when loading it.

Rows after removing duplicates: 302

Rows after removing unknown 'thal': 300

--------------------------------------------------------
Decision Tree Accuracy: 0.80

Random Forest Accuracy: 0.77

---------------------------------------

Cross-validation Accuracy (Decision Tree): 0.7833

Cross-validation Accuracy (Random Forest): 0.8300

--------------------------------------------------
✅ Conclusion:
In this run, the Decision Tree model performed slightly better than the Random Forest model, achieving 3% higher accuracy and more balanced class metrics.
