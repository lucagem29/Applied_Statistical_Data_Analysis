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
