# US Cars Analysis

An interactive data analysis application built with Streamlit to explore a dataset of used cars for sale in the United States.

**Data source:** [Kaggle — US Sales Cars Dataset](https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset)

---

## Features

The app is organized into four sections, navigable from the sidebar:

| Section | Description |
|---|---|
| **Introduction** | Preview the raw dataset, select columns, and inspect data types and null counts |
| **Clean the dataset** | Overview of missing values and the cleaning steps applied |
| **Correlation and links** | Heatmap of numeric correlations and scatter plots between price, mileage, year, brand, and status |
| **Exploratory Data Analysis** | Distribution charts for status, brand, year, average price per brand, most/least sold models, and dealer analysis |
| **Price Prediction** | Random Forest model to predict a car's listing price — shows R², MAE, RMSE, feature importance, and an interactive price estimator |

---

## Getting Started

### Prerequisites

```
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run streamlit_US_cars.py
```

> **Note:** do not run with `python streamlit_US_cars.py` — Streamlit requires its own runtime.

---

## Project Structure

```
US_cars_analysis/
├── streamlit_US_cars.py   # Main application
├── main.ipynb             # Exploratory notebook
└── data/                  # Dataset folder (gitignored)
    └── cars.csv
```

> The `data/` folder is excluded from version control. Download `cars.csv` from the Kaggle link above and place it in `data/` before running the app.

---

## Author

[Alessio Gianello](https://github.com/alessiogianello)