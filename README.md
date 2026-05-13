# 💰 Salary Estimation using KNN Classification

A machine learning project to predict whether a person's income is **<=50K or >50K** based on demographic and work-related features.

---

## 📌 Problem Statement

Predict whether a customer receives a salary of **<=50K or >50K** per year using the KNN (K-Nearest Neighbors) classification algorithm.

---

## 📂 Dataset

- **File:** `salary.csv`
- **Rows:** 32,561
- **Columns:** 5

| Feature | Type | Description |
|---|---|---|
| age | int64 | Age of the person |
| education.num | int64 | Education level (numeric) |
| capital.gain | int64 | Capital gain amount |
| hours.per.week | int64 | Working hours per week |
| income | object | Target: <=50K or >50K |

> ⚠️ **Note:** Dataset is imbalanced — 75% earn <=50K, 25% earn >50K

---

## 🛠️ Technologies Used

- Python 3
- Jupyter Notebook (VS Code)
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Pickle

---

## 🔄 Project Workflow

### 1. 📥 Data Loading & Preprocessing
- Loaded CSV using `pandas`
- Used `LabelEncoder` to convert `income` column → 0 (<=50K) and 1 (>50K)
- Checked for null values — **no missing data found**

### 2. 📊 Exploratory Data Analysis (EDA)
- **Histogram** of age → slightly right-skewed / normal distribution
- **Scatter plot** (age vs hours.per.week) → blue = <=50K, red = >50K
- **Correlation Heatmap** → `education.num` has highest correlation with income (0.34)

### 3. ✂️ Feature Engineering
- Features (X): `age`, `education.num`, `capital.gain`, `hours.per.week`
- Label (y): `income`
- Train/Test Split: **75% / 25%** (`random_state=0`)
- Feature Scaling: `StandardScaler`

### 4. 🔍 Finding Optimal K Value
- Tested `k` from 1 to 40
- Plotted error rate vs k-value
- Error stabilizes around **k=8 to k=15**

### 5. 🤖 Model Training
```python
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2, metric='minkowski', p=2)
model.fit(x_train, y_train)
```

### 6. 📈 Model Evaluation

| Metric | Value |
|---|---|
| ✅ Accuracy | **80.39%** |

**Confusion Matrix:**
```
[[5916  1319]
 [ 277   629]]
```

---

## 🔮 Prediction

The model takes user input and predicts income class:

```python
age = int(input("Enter the customer age: "))
hours = int(input("Enter the customer hours: "))
edu = int(input("Enter the edu: "))
cg = int(input("Enter the cg: "))

newcust = [[age, hours, edu, cg]]
result = model.predict(sc.transform(newcust))

if result == 1:
    print("Customer receives >50K")
else:
    print("Customer receives <=50K")
```

---

## 💾 Model Saving

```python
import pickle
pickle.dump(model, open('model.pkl', 'wb'))
```

---

## 📁 Project Structure

```
salary-estimation-knn/
│
├── salary estimation.ipynb   # Main Jupyter Notebook
├── salary.csv                # Dataset
├── model.pkl                 # Saved KNN model
└── README.md                 # Project documentation
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/yourusername/salary-estimation-knn.git
cd salary-estimation-knn
```

2. Install dependencies:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

3. Open Jupyter Notebook:
```bash
jupyter notebook "salary estimation.ipynb"
```

4. Run all cells!

---

## 💡 Future Improvements

- Handle class imbalance using **SMOTE**
- Try **Random Forest** or **XGBoost** for higher accuracy
- Build a **web app** using Flask or Streamlit
- Use optimal k value (k=10–15) instead of k=2

---

## 👨‍💻 Author

Made with ❤️ as a Machine Learning learning project.
