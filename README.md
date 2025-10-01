# Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📌 Overview

This project builds an **Artificial Neural Network (ANN)** using **TensorFlow/Keras** to predict whether a bank’s customer will **churn** (leave the bank) or **stay**.

The workflow includes:

1. Data preprocessing (cleaning & encoding)
2. Feature scaling
3. Building an ANN
4. Training and evaluating the model
5. Making predictions on new data

---

## 📊 Dataset

The dataset used is **Churn_Modelling.csv**, which contains information about **10,000 bank customers** with the following columns:

* **Customer Information**: RowNumber, CustomerId, Surname
* **Features**: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
* **Target (y)**: `Exited` → (1 if the customer left the bank, 0 otherwise)

---

## 🏷️ Problem Statement

The bank wants to **predict customer churn** in order to:

* Identify at-risk customers
* Improve customer retention
* Reduce financial losses

**Challenge:** Machine learning models can’t process text (like `"France"`, `"Male"`, `"Female"`). We need to preprocess the data into **numerical format** while avoiding mistakes like giving artificial “rankings” to categories.

---

## ⚙️ Data Preprocessing

### 1. Selecting Features

```python
X = dataset.iloc[:, 3:-1].values   # Features
y = dataset.iloc[:, -1].values     # Target (Exited)
```

We exclude **RowNumber, CustomerId, Surname** (not useful for prediction).

---

### 2. Encoding Categorical Data

#### Label Encoding (Gender)

```python
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```

* Converts `["Male", "Female"]` → `[1, 0]`.
* **Problem solved:** ANN can’t handle text; this turns Gender into numbers.

---

#### One-Hot Encoding (Geography)

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```

* Converts `"France"`, `"Spain"`, `"Germany"` into **three binary columns**:

  * France → `[1,0,0]`
  * Spain → `[0,1,0]`
  * Germany → `[0,0,1]`

* **Problem solved:** Avoids giving fake ranking (e.g., France=0, Spain=1, Germany=2).

---

### 3. Splitting Dataset

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

* **80%** training, **20%** testing.

---

### 4. Feature Scaling

```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

* Standardizes data → mean = 0, standard deviation = 1.
* **Problem solved:** Different features (like salary in thousands vs age in tens) are brought to the same scale.

---

## 🧠 Building the ANN

```python
import tensorflow as tf

ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```

* **Input Layer**: Takes all customer features
* **Hidden Layers (2 × 6 neurons, ReLU)**: Learn non-linear patterns
* **Output Layer (1 neuron, Sigmoid)**: Outputs probability of churn (0–1)

---

## ⚡ Compiling & Training

```python
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```

* **Optimizer**: Adam → updates weights efficiently
* **Loss**: Binary Crossentropy → suitable for 0/1 outcomes
* **Metric**: Accuracy
* **Training**: 100 epochs, batch size 32

---

## 🔮 Making Predictions

### Single Customer Prediction

```python
sc.transform([[1,0,0,600,1,40,3,60000,2,1,1,50000]])
ann.predict(...)
```

Predicts if a **40-year-old male from France with a credit score of 600** and balance of 60k will churn.

---

### Test Set Predictions

```python
y_pred = (ann.predict(X_test) > 0.5)
```

Outputs 0/1 predictions for all test customers.

---

## 📈 Model Evaluation

```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```

* **Confusion Matrix**: Shows true vs predicted churn.
* **Accuracy Score**: Overall correctness.

---

## 🧠 Key Learnings

1. **Data preprocessing is critical** → ML models only understand numbers.
2. **Label Encoding vs One-Hot Encoding** → when to use which.
3. **Feature Scaling** ensures fair comparison across features.
4. **ANN structure**: Input → Hidden layers → Output.
5. **Evaluation**: Accuracy and confusion matrix reveal real performance.

---

## 🗂️ Project Structure

```
├── Churn_Modelling.csv   # Dataset
├── churn_ann.py          # Python script with full code
└── README.md             # Project documentation
```

---

## 🚀 Future Improvements

* Use **Dropout layers** to reduce overfitting.
* Try **different optimizers** (RMSprop, SGD).
* Compare with other models (Logistic Regression, Random Forests).
* Deploy model as a **web API**.

