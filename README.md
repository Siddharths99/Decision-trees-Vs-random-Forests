# Decision-trees-Vs-random-Forests
This project compares the performance of Decision Tree and Random Forest classifiers on a heart disease dataset. It includes data cleaning, model training, visualization of the decision tree, feature importance analysis, and cross-validation for reliable performance evaluation.
Heart Disease Prediction – Decision Tree & Random Forest

First 5 Rows:
--------------------------------------------------------
    age  sex  cp  trestbps  chol  fbs  restecg  thalach  exang  oldpeak  slope  ca  thal  target
0   52    1   0       125   212    0        1      168      0      1.0      2   2     3       0
1   53    1   0       140   203    1        0      155      1      3.1      0   0     3       0
2   70    1   0       145   174    0        1      125      1      2.6      0   0     3       0
3   61    1   0       148   203    0        1      161      0      0.0      2   1     3       0
4   62    0   0       138   294    1        1      106      0      1.9      1   3     2       0

Data Cleaning Summary:
--------------------------------------------------------
Rows after removing duplicates: 302
Rows after removing unknown 'thal': 300

Model Performance:
--------------------------------------------------------
Decision Tree Accuracy: 0.80
Decision Tree Classification Report:
               precision    recall  f1-score   support
           0       0.78      0.78      0.78        27
           1       0.82      0.82      0.82        33
    accuracy                           0.80        60
   macro avg       0.80      0.80      0.80        60
weighted avg       0.80      0.80      0.80        60

Random Forest Accuracy: 0.77
Random Forest Classification Report:
               precision    recall  f1-score   support
           0       0.74      0.74      0.74        27
           1       0.79      0.79      0.79        33
    accuracy                           0.77        60
   macro avg       0.76      0.76      0.76        60
weighted avg       0.77      0.77      0.77        60
