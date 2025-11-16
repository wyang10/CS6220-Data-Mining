# Heart Attack Risk Prediction 🫀

Predicting in-hospital mortality for heart attack patients using classical machine learning models and class-imbalance techniques.

This repository contains a reproducible notebook, dataset, and report for a course project. We benchmark six classifiers and evaluate the impact of SMOTE on an imbalanced dataset (approx. 9:91).

Key takeaway: Decision Tree performed best among the baseline models; SMOTE substantially improved minority-class (DIED) recall.

Important note: This project is an academic exercise and is not intended for clinical use.

## Repository Structure

- `CS6220_Data_Mining_Final_Project.ipynb` — main analysis notebook (data prep, modeling, evaluation, visualization)
- `whole_table.csv` — dataset used in the notebook
- `data mining project.pdf`, `CS6220 - Data Mining Final Project.ipynb - Colaboratory.pdf` — writeup and PDF export
- `README.md` — project overview and instructions

## Dataset

Source: Provided as `whole_table.csv` for reproducibility. Columns include:

- `Patient` — anonymized identifier
- `DIAGNOSIS` — diagnosis code
- `SEX` — patient sex (`M`/`F`)
- `DRG` — diagnosis-related group code
- `DIED` — in-hospital mortality outcome (`1` = died, `0` = survived)
- `CHARGES` — total charges (some missing values may be `.`)
- `LOS` — length of stay (days)
- `AGE` — age in years

Class imbalance: The positive class (`DIED=1`) is rare (≈9%).

## Models and Methods

train and compare the following models (scikit-learn):

- Naive Bayes (GaussianNB)
- K-Nearest Neighbors (KNN)
- Decision Tree
- Logistic Regression
- Support Vector Machine (SVM)
- Multi-layer Perceptron (Neural Network)

To address class imbalance, we apply SMOTE (from `imblearn`) and report metrics with and without SMOTE:

- Metrics: training/test accuracy, recall for both classes (with emphasis on `DIED` recall)

## Quick Start

1) Create an environment and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) Launch Jupyter and run the notebook

```bash
jupyter notebook
# Open CS6220_Data_Mining_Final_Project.ipynb and run all cells
```

The notebook now loads the dataset with a repo‑relative path (`whole_table.csv`) so it works out of the box.

Optional (headless execution):

```bash
jupyter nbconvert \
  --to notebook \
  --execute CS6220_Data_Mining_Final_Project.ipynb \
  --output CS6220_Data_Mining_Final_Project.executed.ipynb
```

## Results (Summary)

- Decision Tree achieved the strongest baseline performance on this dataset.
- SMOTE increased minority-class (DIED) recall across multiple models.
- Given the clinical context, prioritize recall for the positive class when selecting models.

For details, see the notebook and the included PDF report.

## Reproducibility Notes

- Random seeds are set in the notebook for model training where applicable.
- If you see parsing issues for `CHARGES` due to `.` values, convert to numeric with `errors="coerce"` and impute or drop as appropriate.

Example snippet:

```python
import pandas as pd
df = pd.read_csv("whole_table.csv")
df["CHARGES"] = pd.to_numeric(df["CHARGES"], errors="coerce")
```

## Disclaimer

This project is for educational purposes only and should not be used for clinical decision‑making.

## 中文简介

这个数据挖掘项目，使用纽约州 1993 年心梗住院患者数据，基于人口学特征与住院信息，比较 6 种经典机器学习模型对住院死亡的预测能力，并评估 SMOTE 在类别不平衡（约 9:91）场景下的效果。主要结论：决策树在基线模型中表现最佳；引入 SMOTE 后，少数类（死亡）召回率明显提升。仓库包含可直接运行的 notebook 与数据集，安装 `requirements.txt` 后在本地即可复现。
