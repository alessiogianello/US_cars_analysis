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

---

## Getting Started

### Prerequisites

```
pip install streamlit pandas matplotlib seaborn numpy folium streamlit-folium
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
├── cars.csv               # Dataset
└── main.ipynb             # Exploratory notebook
```

---

## Author

[Alessio Gianello](https://github.com/alessiogianello)