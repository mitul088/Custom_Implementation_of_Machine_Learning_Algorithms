Note : This repository includes several structured data assets used across different algorithmic benchmarks. This repository serves as a portfolio compilation of machine learning projects I developed locally between December 2023 and October 2024.


# Custom Implementation of Machine Learning Algorithms

A hands-on Jupyter notebook that walks through machine learning from the ground up. You will see the same ideas twice in many places: first built step by step with **NumPy and your own code**, and again using **ready-made tools from scikit-learn** (and sometimes other libraries). That mix is intentional—it helps you understand *what* an algorithm does before you rely on a library to do it for you.

---

## What is in this project?

| Item | Description |
|------|-------------|
| **Main notebook** | `Custom_Implementation_of_Machine_Learning_Algorithms.ipynb` — all code, plots, and explanations |
| **CSV datasets** | Many practice files in the same folder (Titanic, Iris, Placement, etc.) |
| **This README** | A simple map of everything in the notebook |

The notebook is long (hundreds of cells). You do not need to run it all at once. Use the sections below like chapters and jump to what you are learning.

---

## From scratch vs library — how to read the notebook

Throughout the notebook you will notice two styles:

### 1. From scratch (custom code)

You write the math and logic yourself, usually with **NumPy** and **pandas**. Examples include:

- Custom classes such as `SLR`, `MLR`, `GDRN`, `SGDRegressor`, `MBGDR`, `Rid`, `LORFGD`, `MeraRidgeGD`
- Manual steps for **PCA** (scaling → covariance matrix → eigenvalues)
- Commented or partial implementations (e.g. perceptron, gradient boosting for classification)
- Formulas for slope/intercept, gradient descent updates, sigmoid, etc.

**Why this matters:** You see every step—how weights update, how loss goes down, how data is transformed.

### 2. Direct / library methods (scikit-learn and others)

You call built-in classes and functions that do the same job in fewer lines. Examples include:

- `sklearn.linear_model.LinearRegression`, `Ridge`, `Lasso`, `LogisticRegression`
- `sklearn.neighbors.KNeighborsClassifier`
- `sklearn.decomposition.PCA`
- `sklearn.ensemble` (Random Forest, AdaBoost, Bagging, Voting, Stacking, Gradient Boosting)
- `sklearn.cluster` (K-Means, Agglomerative, DBSCAN)
- `sklearn.preprocessing`, `sklearn.pipeline`, `GridSearchCV`, and more

**Why this matters:** This is what you typically use in real projects—fast, tested, and consistent APIs (`fit`, `predict`, `transform`).

### How to use both together

| When you are… | Focus on… |
|---------------|-----------|
| Learning a topic for the first time | From-scratch cells, then sklearn version right after |
| Building a project quickly | Library methods |
| Debugging or interviews | From-scratch understanding |

If a section only shows sklearn, look backward in the notebook—there is often a manual version nearby. If code is **commented out** (e.g. perceptron from scratch), uncomment and run it to compare with the library result.

---

## What you need before starting

### Software

- **Python 3.8+** (3.10 or 3.11 recommended)
- **Jupyter Notebook** or **JupyterLab** (or VS Code / Cursor with a notebook extension)

### Python libraries

Install the packages used in the notebook:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy plotly mlxtend xgboost
```

| Library | Used for |
|---------|----------|
| `numpy` | Arrays, linear algebra, custom algorithms |
| `pandas` | Loading CSVs, cleaning, exploring data |
| `matplotlib` / `seaborn` | Charts and exploratory plots |
| `scikit-learn` | Most ML models, preprocessing, metrics, pipelines |
| `scipy` | Extra scientific utilities where needed |
| `plotly` | Interactive plots (some cells) |
| `mlxtend` | Optional plotting helpers (some ensemble examples) |
| `xgboost` | XGBoost section near the end |

### How to run

1. Open a terminal in this folder (where the `.ipynb` and `.csv` files live).
2. Start Jupyter: `jupyter notebook` or `jupyter lab`.
3. Open `Custom_Implementation_of_Machine_Learning_Algorithms.ipynb`.
4. Run cells **top to bottom** within each section, or use **Run All** only when you have time and enough RAM (the file is large).

Keep all CSV files in the **same directory** as the notebook so `pd.read_csv('filename.csv')` works without path changes.

---

## Notebook roadmap (all major topics)

Below is everything covered, in roughly the order it appears. Section names match headings or comments inside the notebook.

### Part 1 — First model and data exploration

| Topic | What you learn |
|-------|----------------|
| **Student placement prediction** | Load `Placement.csv`, plot CGPA vs IQ, train/test split, **LogisticRegression** (sklearn), accuracy |
| **Titanic — preprocessing & training** | Missing values, encoding, building a classifier on real messy data |
| **Exploratory data analysis (EDA)** | Histograms, bivariate plots, pairplots (Iris, Titanic, etc.) with **seaborn** |

### Part 2 — Preparing data for ML

| Topic | From scratch / manual | Library (sklearn) |
|-------|-------------------------|-------------------|
| **Feature scaling** | Concepts of scale | `StandardScaler`, `MinMaxScaler` |
| **Normalization** | When and why | Same scalers, applied in pipelines |
| **Encoding categories** | Ideas behind one-hot | `OrdinalEncoder`, `LabelEncoder`, `OneHotEncoder` (including k-1 encoding notes) |
| **Column transformer** | Applying different steps per column | `ColumnTransformer` |
| **Pipelines** | Chaining steps by hand | `Pipeline`, `make_pipeline` |
| **Missing values** | Simple fill strategies | `SimpleImputer`, `MissingIndicator`, **KNN imputer** |
| **Outlier detection & removal** | Z-score, trimming, IQR/boxplot, winsorization/capping | Often pandas + rules; concepts apply before modeling |

Datasets used in this part include **Titanic**, **Social Network Ads**, **wine**, **heart**, **customer**, and others.

### Part 3 — Dimensionality reduction (PCA)

| Step | Approach |
|------|----------|
| Standardize features | Manual + `StandardScaler` |
| Covariance matrix | NumPy |
| Eigenvalues / principal components | Manual math + `PCA` from sklearn |
| Transform and visualize | Compare manual vs `PCA(n_components=...)` |

### Part 4 — Regression

| Algorithm | From scratch | Library |
|-----------|--------------|---------|
| **Simple linear regression (SLR)** | Class `SLR` (slope & intercept formulas) | `LinearRegression` |
| **Multiple linear regression (MLR)** | Class `MLR` | `LinearRegression` |
| **Gradient descent (batch)** | Class `GDRN`, `FindingB` | Compare to closed-form / sklearn |
| **Stochastic GD (SGD)** | Class `SGDRegressor` | `sklearn.linear_model.SGDRegressor` |
| **Mini-batch GD** | Class `MBGDR` | Same idea, different batch size |
| **Polynomial regression** | Manual features & fitting | `PolynomialFeatures` + `LinearRegression` |
| **3D / multivariate polynomial** | Extended examples | sklearn pipeline style |
| **Ridge (L2)** | Classes `Rid`, `RidN`, `MeraRidgeGD`, ridge from scratch | `Ridge`, cholesky solver |
| **Lasso (L1)** | Concepts | `Lasso` |
| **SVR** | — | `sklearn.svm.SVR` |

Common dataset: **diabetes** (`load_diabetes`), **Boston housing**, **placement** files, synthetic data.

### Part 5 — Classification

| Algorithm | From scratch | Library |
|-----------|--------------|---------|
| **Logistic regression** | Gradient descent class `LORFGD`, sigmoid | `LogisticRegression` |
| **Perceptron** | Commented scratch implementation | sklearn / linear models related material |
| **KNN** | Distance idea (conceptually) | `KNeighborsClassifier` |
| **Decision tree** | How splits work (intuition in notebook) | `DecisionTreeClassifier`, `DecisionTreeRegressor`, `plot_tree` |

Synthetic data from `make_classification`; **Iris** CSV for ensemble demos.

### Part 6 — Ensemble learning

| Method | What it does | In notebook |
|--------|--------------|-------------|
| **Voting** | Combine several models’ votes | `VotingClassifier`, `VotingRegressor` |
| **Bagging** | Many models on bootstrap samples | `BaggingRegressor`, bagged trees |
| **Random Forest** | Many randomized trees | `RandomForestClassifier` |
| **AdaBoost** | Weighted weak learners | `AdaBoostClassifier` |
| **Gradient Boosting** | Sequential error correction | Scratch notes + `GradientBoostingClassifier` |
| **Stacking** | Meta-model on base models | `StackingClassifier` |
| **Grid Search CV** | Tune hyperparameters | `GridSearchCV`, `cross_val_score` |

### Part 7 — Clustering (unsupervised)

| Algorithm | Library / notes |
|-----------|-----------------|
| **K-Means** | `KMeans` — 2D and 3D examples, `make_blobs` |
| **Agglomerative (hierarchical)** | `AgglomerativeClustering` — e.g. shopping data |
| **DBSCAN** | Density-based clusters, noise points |
| **XGBoost** | Separate section (`xgboost`) — often used for structured tabular data |

Datasets: **student_clustering**, **shoppingdata_hierarchical**, synthetic circles/blobs.

---

## Datasets included in this folder

The notebook loads different CSVs for different lessons. Keep these files next to the notebook:

- `Placement.csv`, `placement (1).csv`, `placement (2).csv`
- `titanic.csv`, `train.csv`
- `iris.csv`
- `Social_Network_Ads.csv`
- `BostonHousing.csv`
- `wine.csv`, `heart1.csv`, `customer.csv`
- `student_clustering.csv`, `shoppingdata_hierarchical.csv`
- `cars.csv`, `covid_toy.csv`, `weight-height.csv`
- `PlayTennis.csv`, `data_science_job.csv`
- And others in the same directory

Some cells use **built-in sklearn datasets** (`load_iris`, `load_diabetes`, `make_blobs`, `make_circles`, etc.) so no CSV is required for those parts.

---

## Custom classes defined in the notebook (quick reference)

These are your **from-scratch** building blocks. Each usually has `fit` and/or `predict` similar to sklearn:

| Class | Purpose |
|-------|---------|
| `SLR` | Simple linear regression (one feature) |
| `MLR` | Multiple linear regression |
| `FindingB` | Helper for gradient descent |
| `GDRN` | Batch gradient descent regressor |
| `SGDRegressor` | Stochastic gradient descent (custom, not sklearn’s class name) |
| `MBGDR` | Mini-batch gradient descent |
| `Rid` / `RidN` | Ridge regression variants |
| `MeraRidgeGD` | Ridge with gradient descent |
| `LORFGD` | Logistic regression with gradient descent |

After training a custom model, compare predictions and metrics with the matching **sklearn** estimator on the same train/test split.

---

## Tips for beginners

1. **Run one section at a time** — The notebook is a course, not a single script.
2. **Read comments** — Lines starting with `#` often say “from scratch” or explain the next library call.
3. **Watch for train/test split** — Almost every model section uses `train_test_split`; always fit on train, evaluate on test.
4. **Same random_state** — When you compare scratch vs sklearn, use the same `random_state` in `train_test_split` for a fair comparison.
5. **Large outputs** — Some cells produce big plots or HTML; that is normal. Clear outputs if the file feels slow.
6. **Typos in filenames** — Some CSV names have spaces or variants (e.g. `Placement (2).csv`). Match the name exactly as in `read_csv`.

---

## What you will be able to do after working through this

- Clean and prepare tabular data (scaling, encoding, imputation, outliers)
- Explain how **linear models**, **PCA**, **KNN**, **trees**, and **ensembles** work under the hood
- Implement core ideas in **NumPy** and use **scikit-learn** for production-style workflows
- Tune models with **cross-validation** and **grid search**
- Apply **clustering** and compare **K-Means**, **hierarchical**, and **DBSCAN**

---

## Project structure

```
Custom Implementation of Machine Learning Algorithms/
├── README.md                                          ← you are here
├── Custom_Implementation_of_Machine_Learning_Algorithms.ipynb
├── Placement.csv
├── titanic.csv
├── iris.csv
├── … (other .csv datasets)
```

---

## License and usage

This repository is for **learning and practice**. Datasets are standard teaching datasets from various sources; use them only for education unless you have rights for other purposes.

---

## Summary

| Question | Answer |
|----------|--------|
| What is this? | One big notebook teaching ML algorithms |
| Why two styles? | **Scratch** = understanding; **sklearn** = speed and best practice |
| Where do I start? | Open the `.ipynb`, begin with *Student Placement Prediction*, then follow the roadmap above |
| What if I only want libraries? | Jump to cells that import from `sklearn` |
| What if I only want math/code? | Focus on custom `class` definitions and PCA/regression scratch sections |

Happy learning — build it yourself once, then let the library do the heavy lifting when you need to.
