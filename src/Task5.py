import pandas as pd
import numpy as np
import graphviz
import matplotlib.pyplot as plt

#Loading and cleaning Dataset
df = pd.read_csv("heart.csv")
print("First 5 rows:\n", df.head())

df = df.drop_duplicates() 
print("\nRows after removing duplicates:", df.shape[0])

if (df['thal'] == 0).any():
    df = df[df['thal'] != 0]
    print("Rows after removing unknown 'thal':", df.shape[0])

#Data splitting
from sklearn.model_selection import train_test_split, cross_val_score
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("\nDecision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("\nDecision Tree Classification Report:\n", classification_report(y_test, y_pred_dt))

#Decision Tree Visualization
plt.figure(figsize=(18,10))
plot_tree(dt, feature_names=X.columns, class_names=['No Disease', 'Disease'], 
          filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=4)")
plt.show()


dot_data = export_graphviz(dt, out_file=None, feature_names=X.columns,
            class_names=['No Disease', 'Disease'], filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree_graphviz", format="png", cleanup=True)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nRandom Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("\nRandom Forest Classification Report:\n", classification_report(y_test, y_pred_rf))

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10,6))
import seaborn as sns
sns.barplot(x=importances, y=importances.index, hue=importances.index, palette="viridis", legend=False)
plt.title("Random Forest Feature Importances")
plt.show()

#Cross-validation scores
dt_cv = cross_val_score(dt, X, y, cv=5).mean()
rf_cv = cross_val_score(rf, X, y, cv=5).mean()
print(f"\nCross-validation Accuracy (Decision Tree): {dt_cv:.4f}")
print(f"Cross-validation Accuracy (Random Forest): {rf_cv:.4f}")

#Model comparsion
if accuracy_score(y_test, y_pred_rf) > accuracy_score(y_test, y_pred_dt):
    print("\nRandom Forest performed better on the test set.")
else:
    print("\nDecision Tree performed better on the test set.")