# Disease Classification – MLOps Project

## Project Overview

This project implements an **end-to-end MLOps pipeline** for deploying a **CatBoost-based classification model** that predicts the **clinical status of patients with Primary Biliary Cirrhosis (PBC)**.

The model is trained and evaluated using the **Primary Biliary Cirrhosis dataset from the :contentReference[oaicite:0]{index=0}**, a well-known dataset in medical research.  
The project follows **industry-grade MLOps best practices**, focusing on reproducibility, automation, scalability, and reliability.

---

## Objectives

- Build a **production-ready ML pipeline**
- Ensure **experiment reproducibility**
- Track experiments and models with **MLflow**
- Version datasets with **DVC**
- Automate workflows using **GitHub Actions**
- Enable cloud deployment with **AWS**
- Enforce code quality via **unit testing**

---

## Project Organization

```
disease_classification/
├── components/          # Core ML pipeline logic
│   ├── prepare.py       # Data preprocessing
│   ├── training.py      # Model training
│   └── evaluate.py      # Model evaluation
│
├── configs/             # YAML configuration files
│   ├── prepare.yaml
│   ├── training.yaml
│   ├── evaluate.yaml
│   └── prediction.yaml
│
├── data/
│   └── raw/             # Datasets tracked with DVC
│       ├── train.csv
│       ├── test.csv
│
├── models/              # Saved trained models
│
├── notebooks/
│   └── Main.ipynb       # EDA and experimentation
│
├── src/                 # Executable pipeline scripts
│   ├── prepare.py
│   ├── training.py
│   ├── evaluate.py
│   └── prediction.py
│
├── tests/               # Unit tests
│   ├── test_prepare.py
│   ├── test_training.py
│   ├── test_evaluate.py
│   └── test_utils.py
│
├── utils/               # Shared utilities
│   └── common.py
│
├── dockerfile           # Docker image definition
├── pyproject.toml       # Project dependencies
├── uv.lock              # Dependency lock file
└── README.md
```

---

## License

This project is for personal and educational purposes only.
All rights reserved.

--------

