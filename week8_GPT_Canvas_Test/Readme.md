# **Machine Learning Model Comparison Framework**

This project provides a reusable framework to compare machine learning models for classification tasks. It focuses on Logistic Regression, Generalized Linear Models (GLMs), and Decision Trees, along with baseline evaluation using a Dummy Classifier. The framework also includes model evaluation metrics, hyperparameter tuning, and visualizations.

---

## **Table of Contents**

1. [Project Description](#project-description)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Notebook Structure](#notebook-structure)
7. [Supported Models](#supported-models)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Future Enhancements](#future-enhancements)
10. [License](#license)

---

## **Project Description**

This framework allows users to:
- Load and preprocess datasets (scaling numeric variables, encoding categorical variables).
- Train baseline and advanced models for classification tasks.
- Evaluate models using various performance metrics.
- Compare model performance using ROC curves, confusion matrices, and statistical measures.
- Visualize results for better interpretability.

The project is implemented in a Jupyter Notebook, designed for flexibility and interactivity.

---

## **Features**

1. **Baseline Evaluation**:
   - Dummy Classifier to establish a baseline for model comparison.

2. **Advanced Models**:
   - Logistic Regression
   - Generalized Linear Models (GLMs)
   - Decision Trees

3. **Evaluation Metrics**:
   - Accuracy, F1-Score, ROC AUC, MCC, and more.

4. **Visualizations**:
   - Confusion Matrices
   - ROC Curves

5. **Hyperparameter Tuning**:
   - Built-in support for Grid Search to optimize Decision Trees and other models.

6. **Reusability**:
   - Modular functions for loading data, preprocessing, training, and evaluation.

---

## **Prerequisites**

- Python 3.8 or later
- Jupyter Notebook installed
- Required Python libraries:
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `statsmodels`
  - `matplotlib`
  - `seaborn`

---

## **Installation**

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>

pip install -r requirements.txt


jupyter notebook

## **Usage**

### **1. Prepare Your Dataset**:
- Save your dataset as a CSV file (e.g., `your_dataset.csv`).
- Ensure the target column is labeled (e.g., `target`).

### **2. Run the Notebook**:
- Open `ML_Model_Comparison_Notebook_EN.ipynb` in Jupyter Notebook.
- Update the dataset path and target column in the `Load and Prepare Data` section:
  ```python
  dataset_path = "your_dataset.csv"
  target_column = "target"
  ```

### **3. Follow the Steps**:
- Execute cells sequentially to preprocess the data, train models, and visualize results.

### **4. Analyze Results**:
- Compare models using metrics and visualizations such as ROC curves and confusion matrices.

---

## **Notebook Structure**

### **Introduction**:
- Overview of the framework.

### **Data Loading and Preprocessing**:
- Functions to load and preprocess datasets.

### **Baseline Evaluation**:
- Dummy Classifier as a benchmark.

### **Model Training and Evaluation**:
- Training Logistic Regression, GLMs, and Decision Trees.
- Generating evaluation metrics.

### **Visualization**:
- Plotting confusion matrices and ROC curves.

### **Advanced Features**:
- Hyperparameter tuning for Decision Trees.

---

## **Supported Models**

### **Baseline Model**:
- Dummy Classifier (`sklearn.dummy.DummyClassifier`)

### **Classification Models**:
- Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- Generalized Linear Model (GLM, via `statsmodels`)
- Decision Tree (`sklearn.tree.DecisionTreeClassifier`)

---

## **Evaluation Metrics**

### **Model Performance**:
- Accuracy
- F1-Score
- ROC AUC (Receiver Operating Characteristic Area Under the Curve)
- Matthews Correlation Coefficient (MCC)

### **Statistical Measures (for GLMs)**:
- AIC (Akaike Information Criterion)
- BIC (Bayesian Information Criterion)

### **Visualization**:
- Confusion Matrices
- ROC Curves

---

## **Future Enhancements**

- Add support for additional models:
  - Random Forest
  - Support Vector Machines (SVM)
- Automate feature selection for improved performance.
- Extend functionality for multiclass classification.
- Implement SHAP for explainable AI insights.

---

## **License**

This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## **Acknowledgements**

This project uses popular Python libraries such as `scikit-learn`, `statsmodels`, and `seaborn` for model building and visualizations.
