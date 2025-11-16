<!--
 * @Author: Audrey Yang 97855340+wyang10@users.noreply.github.com
 * @Date: 2025-11-15 15:41:34
 * @LastEditors: Audrey Yang 97855340+wyang10@users.noreply.github.com
 * @LastEditTime: 2025-11-15 18:07:35
 * @FilePath: /Smote-Heart-Attack-ML/README-1.md
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
-->
                   📊 Heart Attack ML Pipeline (Architecture Overview)
                   ---------------------------------------------------
       ┌───────────────────────────────────────────────────────────────────┐
       │                           DATA LAYER                              │
       └───────────────────────────────────────────────────────────────────┘
                            Raw CSV (whole_table.csv)
                                       │
                                       ▼
                        src/data_loader.py  → minimal cleaning
                         - CHARGES coercion
                         - dtype fixing
                         - missing drop/clean
       ┌───────────────────────────────────────────────────────────────────┐
       │                       FEATURE ENGINEERING                         │
       └───────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                           src/features.py
                         - Train/Test Split
                         - OHE for categoricals
                         - Scaling numeric fields
                         - Optional SMOTE (--smote)
       ┌───────────────────────────────────────────────────────────────────┐
       │                        MODELING LAYER                             │
       └───────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                      src/models.py   → Model Factory
                    -------------------------------------
                    |  Naive Bayes       |  KNN          |
                    |  Decision Tree     |  LogisticReg  |
                    |  SVM               |  MLP          |
                    -------------------------------------
                     Select model via CLI flag: --model <name>
       ┌───────────────────────────────────────────────────────────────────┐
       │                         TRAINING / EVAL                           │
       └───────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
                 scripts/train.py                  scripts/evaluate_model.py
                 - fit model                       - load model
                 - compute metrics                 - generate metrics.json
                 - save artifacts                  - print recall/accuracy
       ┌───────────────────────────────────────────────────────────────────┐
       │                       OUTPUT ARTIFACTS                            │
       └───────────────────────────────────────────────────────────────────┘
                 data/processed/
                 ├── trained_model.joblib       (final model)
                 ├── scaler.joblib              (for numeric fields)
                 └── metrics.json               (recall, accuracy, reports)
       ┌───────────────────────────────────────────────────────────────────┐
       │                           NOTEBOOK                                │
       └───────────────────────────────────────────────────────────────────┘
                 notebooks/heart_attack_eda.ipynb
                 - EDA
                 - feature exploration
                 - visualizations (CM, ROC)

---

> **DATA LAYER**
>
> 1. Raw CSV (`whole_table.csv`)
> 2. **src/data_loader.py** → minimal cleaning
>     * CHARGES coercion
>     * dtype fixing
>     * missing drop/clean
>   
> **FEATURE ENGINEERING**
>
> 3. **src/features.py**
>     * Train/Test Split
>     * OHE for categoricals
>     * Scaling numeric fields
>     * Optional SMOTE (`--smote`)
>
> **MODELING LAYER**
>
> 4. **src/models.py** → Model Factory
>     * Naive Bayes, KNN, Decision Tree, Logistic Regression, SVM, MLP
>     * Select model via CLI flag: `--model <name>`
>
> **TRAINING / EVAL**
>
> 5. **scripts/train.py** & **scripts/evaluate_model.py**
>     * fit model, compute metrics, save artifacts
>
> **OUTPUT ARTIFACTS**
>
> 6. **data/processed/**
>     * `trained_model.joblib` (final model)
>     * `scaler.joblib` (for numeric fields)
>     * `metrics.json` (recall, accuracy, reports)
>
> **NOTEBOOK**
>
> 7. **notebooks/heart_attack_eda.ipynb**
>     * EDA, feature exploration, visualizations (CM, ROC)