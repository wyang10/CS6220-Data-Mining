# Heart Attack Risk

Modular, reproducible project for predicting in-hospital mortality of heart attack patients. Includes EDA notebook, data loaders, feature engineering, multiple models, and CLI scripts for training and evaluation.

## Structure

```
heart-attack-risk/
├─ data/
│  ├─ raw/              # 原始CSV（可忽略提交）
│  └─ processed/        # 清洗/中间数据与模型输出
├─ notebooks/
│  └─ heart_attack_eda.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ config.py         # 常量/路径
│  ├─ data_loader.py    # 读写数据
│  ├─ features.py       # 特征工程
│  ├─ models.py         # 模型定义/训练
│  └─ evaluate.py       # 评估与指标
├─ scripts/
│  ├─ train.py          # 命令行入口：训练
│  └─ evaluate_model.py # 命令行入口：评估
├─ requirements.txt
├─ README.md
└─ .gitignore
```

## Quickstart

1) 安装依赖（建议虚拟环境）

```
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2) 放置原始数据

- 将 `whole_table.csv` 放入 `data/raw/`（本仓库示例已放好）

3) 训练模型（例：决策树 + SMOTE）

在心态一致的目录下执行有三种等价方式：

```
# 方式A（推荐）：先进入项目目录
cd heart-attack-risk
python -m scripts.train --model decision_tree --smote

# 方式B：从仓库根目录直接执行脚本
python heart-attack-risk/scripts/train.py --model decision_tree --smote

# 方式C：设置 PYTHONPATH 后使用 -m
PYTHONPATH=heart-attack-risk python -m scripts.train --model decision_tree --smote
```

输出：
- 模型：`data/processed/trained_model.joblib`
- 指标：`data/processed/metrics.json`

4) 评估已保存模型

```
# 方式A
cd heart-attack-risk && python -m scripts.evaluate_model --model-path data/processed/trained_model.joblib

# 方式B
python heart-attack-risk/scripts/evaluate_model.py --model-path heart-attack-risk/data/processed/trained_model.joblib

# 方式C
PYTHONPATH=heart-attack-risk python -m scripts.evaluate_model --model-path heart-attack-risk/data/processed/trained_model.joblib
```

## Notebook

- `notebooks/heart_attack_eda.ipynb` 演示完整流程（数据预处理、建模、可视化）。
- 已将数据路径设置为相对路径：`../data/raw/whole_table.csv`。

## 说明

- 类别不平衡：数据中 `DIED=1` 占比约 9%，可通过 `--smote` 启用 SMOTE 提升少数类召回。
- 字段：`DIAGNOSIS, SEX, DRG, AGE, LOS, CHARGES` 作为特征（其中 `CHARGES` 自动转为数值类型）。
- 结果侧重于少数类（死亡）的召回率与宏平均 F1。

## 免责声明

本项目为教学用途，非临床决策依据。
