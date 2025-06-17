# 💳 Credit Card Fraud Detection using Decision Trees and Random Forests

This project implements machine learning models to detect fraudulent transactions in a real-world credit card dataset using scikit-learn. It focuses on model performance **and interpretability**, especially in imbalanced classification settings relevant to fraud and risk advisory.

---

## 📊 Dataset

- Source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions
- 492 fraud cases (high class imbalance)
- 30 numerical features (`V1–V28` are PCA components, `Amount`, `Time`)
- Target: `Class` (0 = Not Fraud, 1 = Fraud)

---

## ⚙️ Approach

1. **Preprocessing**
   - Stratified train-test split to preserve class distribution

2. **Modeling**
   - Trained a baseline `DecisionTreeClassifier`
   - Applied **cost-complexity pruning** using cross-validated `ccp_alpha`

3. **Evaluation**
   - Focused on **precision, recall, F1-score**, and **AUC** due to data imbalance
   - Plotted and interpreted the confusion matrix
   - Visualized the final pruned decision tree

---

## 📈 Results

| Metric (Fraud Class) | Before Pruning | After Pruning |
|----------------------|----------------|----------------|
| Precision            | 0.75           | **0.92**       |
| Recall               | 0.74           | **0.78**       |
| F1-Score             | 0.75           | **0.84**       |

- Tree size reduced by ~60%, improving interpretability
- Maintained strong predictive performance

---

## 🔍 Key Learnings

- How to build interpretable models for fraud detection in imbalanced datasets
- Practical use of pruning (`ccp_alpha`) to avoid overfitting
- Importance of precision and recall in risk-focused ML applications

---

## 📁 Project Files

- `fraud_detection.ipynb`: Main notebook with modeling pipeline
- `decision_tree.png`: Visualization of pruned decision tree
- `confusion_matrix.png`: Evaluation output

---

## 🤝 Relevance

This project is highly applicable to **risk advisory, audit analytics**, or **machine learning for compliance and fraud**.

---

## 📬 Contact

Feel free to reach out if you’d like to collaborate or have questions about this project!

