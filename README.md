# 🏦 Loan Status Prediction: Random Forest Classifier

## Project Overview

This project focuses on building a machine learning model to predict the **Loan Status** (Approved: **Y** or Rejected: **N**) based on various applicant features such as income, credit history, education, and property area.

We utilize the **Random Forest Classifier**, an ensemble learning method known for its high accuracy, stability, and ability to handle non-linear relationships and interactions between features. This is a critical business problem for any lending institution to manage risk and process applications efficiently.

## Key Files

| File Name | Description |
| :--- | :--- |
| `Loans_prediction_rf.ipynb` | The main Jupyter Notebook containing data preprocessing, feature engineering, Random Forest model training, hyperparameter tuning (if applicable), and evaluation. |
| `loan_train_data.csv` | The primary dataset used for training and testing the model. |
| `README.md` | This overview file. |

## Methodology

### 1. Data Preprocessing & Feature Engineering
The initial step involved preparing the data, which often contains missing values and categorical data:

* **Handling Missing Values:** Missing values in critical columns like `LoanAmount`, `Credit_History`, and `Gender` were imputed (filled in) using appropriate strategies (e.g., mean imputation, mode imputation).
* **Feature Encoding:** All nominal categorical features (like `Gender`, `Married`, `Education`, `Property_Area`) were converted into numerical features using techniques like Label Encoding or One-Hot Encoding.
* **Data Splitting:** The dataset was divided into training and testing sets to validate the model on unseen data.

### 2. Random Forest Classifier
The Random Forest model was chosen for its ability to combine multiple decision trees to produce a more robust and accurate prediction.

* **Model Training:** The classifier was trained on the preprocessed training data.
* **Feature Importance:** As an added benefit, the Random Forest model provided insights into the most influential factors determining loan status (e.g., typically `Credit_History` and `ApplicantIncome`).

## Model Performance Analysis

The model achieved high overall performance, though a deeper look at the class-specific metrics is crucial for a financial risk problem:

| Metric | Overall Result |
| :--- | :--- |
| **Accuracy (Weighted Avg)** | $\approx \mathbf{0.85}$ |
| **Precision (Weighted Avg)** | $\mathbf{0.86}$ |
| **Recall (Weighted Avg)** | $\mathbf{0.85}$ |
| **F1-Score (Weighted Avg)** | $\mathbf{0.83}$ |

### Class-Specific Analysis

| Loan Status | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- |
| **N (Rejected)** | $\mathbf{0.91}$ | $\mathbf{0.55}$ | $\mathbf{0.69}$ |
| **Y (Approved)** | $\mathbf{0.83}$ | $\mathbf{0.98}$ | $\mathbf{0.90}$ |

#### Interpretation:

1.  **Risk Management (Class N - Rejections):**
    * **High Precision ($\mathbf{0.91}$):** The model is very accurate when it *predicts* a loan will be rejected. This is excellent for risk control, as few loans that the model flags for rejection are actually good candidates.
    * **Low Recall ($\mathbf{0.55}$):** This is the major area for improvement. $45\%$ of loans that *should have been rejected* were mistakenly approved (False Negatives). **This represents high potential financial risk** and indicates the model needs to be tuned to be more sensitive to rejection criteria.

2.  **Customer Service (Class Y - Approvals):**
    * **High Recall ($\mathbf{0.98}$):** The model successfully identifies almost all (98%) of the good loan candidates. This is great for minimizing lost business opportunities.

**Conclusion:**

The Random Forest model is a powerful predictor with an overall accuracy around $\mathbf{85\%}$. While it is exceptionally good at finding approvable loans, the relatively low Recall for the 'Rejected (N)' class suggests that future work should focus on **reducing False Negatives** (missed rejections) to mitigate potential financial loss. Techniques like balancing the dataset (SMOTE) or hyperparameter tuning the Random Forest (e.g., adjusting class weights) could improve the Recall for the minority class.

## Technologies and Libraries

* **Python 3.x**
* **VS code**
* `pandas` (for data manipulation)
* `scikit-learn` (for **RandomForestClassifier**, preprocessing, and metrics)

You can also view the notebook directly on GitHub or platforms like **Google Colab** without needing a local setup.

## 👩‍💻 Author
**Aditi K**  
CSE (AI & ML) | Loan Prediction | RF
