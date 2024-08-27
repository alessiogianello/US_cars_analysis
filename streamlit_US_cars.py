import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pandas as pd
import os
import numpy as np

import folium
import io
from io import BytesIO
from streamlit_folium import st_folium

# Dataset Import
# Load the dataset with UTF-16 encoding into a DataFrame.
cars_list_df = pd.read_csv("cars.csv", encoding="utf-16")


# Define the names of the tabs to be displayed in the sidebar.
tab_names = [
    "Introduction",
    "Clean the dataset",
    "Correlation and links",
    "Exploratory Data Analysis",
]

# Create a sidebar with a selectbox for choosing different tabs.
current_tab = st.sidebar.selectbox("Table of content", tab_names)
st.sidebar.markdown(
    """
    **My GitHub page:**   [GitHub](https://github.com/alessiogianello)  
    """
)


# Function to clean the dataset
def clean(df):

    clean_df = df.copy()

    # correctly format columns names
    clean_df.columns = clean_df.columns.map(lambda x: x.lower())

    # fill null values where possible
    clean_df["mileage"].fillna(0, inplace=True)
    clean_df["dealer"].fillna("unkown_dealer", inplace=True)

    # dropna 'price'
    clean_df = clean_df.dropna(subset=["price"])

    return clean_df


# Introduction
if current_tab == "Introduction":
    # Display introductory text with titles centered.
    st.markdown(
        "<h1 style='text-align: center;'>Exploring the Data</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<h2 style='text-align: center;'>Data analysis project</h2>",
        unsafe_allow_html=True,
    )
    st.markdown(
        """
    This dataset provides comprehensive information about used cars available for sale in the United States. \n
     **Data source:** https://www.kaggle.com/datasets/juanmerinobermejo/us-sales-cars-dataset
    """
    )
    # Allow users to select columns to display.
    selected_columns = st.multiselect(
        "Explore the dataset by selecting columns", cars_list_df.columns
    )

    # Display the selected columns or the first 15 rows if none are selected.
    if selected_columns:
        columns_df = cars_list_df.loc[:, selected_columns]
        st.dataframe(columns_df.head(15))
    else:
        st.dataframe(cars_list_df.head(15))

    st.write("General informations about the DataFrame")
    # Capture information about the DataFrame structure and display it.
    buffer = io.StringIO()
    cars_list_df.info(buf=buffer)
    s = buffer.getvalue()

    # Allow users to select specific columns for detailed information.
    info_columns = st.multiselect(
        "Select the variables",
        cars_list_df.columns.tolist(),
        default=cars_list_df.columns.tolist(),
    )

    # Display info only about the selected columns.
    if info_columns:
        selected_info_buffer = io.StringIO()
        cars_list_df[info_columns].info(buf=selected_info_buffer)
        selected_info = selected_info_buffer.getvalue()
        st.text(selected_info)
    else:
        # If no specific columns are selected, show info for all columns.
        st.text(s)

# Data Cleaning Tab

elif current_tab == "Clean the dataset":
    st.title("Cleaning NA values")

    # Clean the dataset using the previously defined function.
    clean_df = clean(cars_list_df)

    # Create tabs for viewing missing values and cleaned data.
    tab1, tab2, tab3, tab4 = st.tabs(["NA values", "Cleaning", "-", "-"])

    with tab1:
        # Calculate and display the count and percentage of missing values for each variable.
        missing_values_count = cars_list_df.isna().sum()
        total_values = cars_list_df.shape[0]
        missing_values_percentage = (missing_values_count / total_values) * 100
        missing_values_percentage = missing_values_percentage.round(2)

        # Create a DataFrame to show missing values and their percentages.
        missing_df = pd.DataFrame(
            {
                "Variable": missing_values_count.index,
                "NA values": missing_values_count.values,
                "%  NA values": missing_values_percentage.values,
            }
        )

        st.write(missing_df)

    with tab2:
        # Explain the cleaning process and show a preview of the cleaned DataFrame.
        st.markdown(
            """
                Based on these information the data will be rearranged as follows:
                - **fixed** the column indexes
                - **replaced** the NaN values in the columns 'dealership' with 'unknown_delaership'
                - **removed** values with NaN price
                
                                """
        )
        with st.expander("Resulted DataFrame Preview"):
            st.write(clean_df.head(15))

elif current_tab == "Correlation and links":
    st.title("Correlation and links")

    # Clean the dataset.
    clean_df = clean(cars_list_df)

    # Calculate correlations between numeric variables.
    numeric_df = cars_list_df_corr = cars_list_df.corr(numeric_only=True)

    # Display insights on correlations observed in the data.
    st.markdown(
        """
        I see that Year and Mileage have a decent negative correlation (and it makes sense: newer the car, lower the milege), 
        furthermore I see a small correlation between year and price, and as it may be expected i see that mileage and 
        price are negatively correlated.
        """
    )

    # Create a heatmap to visualize correlations.
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        numeric_df,
        annot=True,
        cmap="BuPu",
    )

    st.pyplot(plt.gcf())

    st.markdown(
        """
        Here some interesting scatter plots:
        """
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Mileage vs Year",
            "Mileage vs Price",
            "Price vs Year",
            "Price vs Mileage vs Status",
            "Price vs Mileage vs Brand",
        ]
    )

    with tab1:
        # Scatter plot of Mileage vs Year
        plt.figure(figsize=(8, 6))
        sns.set_palette("husl")
        sns.scatterplot(data=clean_df, x="mileage", y="year")
        plt.xlabel("Mileage")
        plt.ylabel("Year")
        plt.title("Relationship between mileage and year")
        st.pyplot(plt.gcf())

    with tab2:
        # Scatter plot of Mileage vs Price
        plt.figure(figsize=(8, 6))
        sns.set_palette("husl")
        sns.scatterplot(data=clean_df, x="mileage", y="price")
        plt.xlabel("Mileage")
        plt.ylabel("Price")
        plt.ticklabel_format(style="plain", axis="y")
        plt.title("Relationship between mileage and price")
        st.pyplot(plt.gcf())

    with tab3:
        # Scatter plot of Price vs Year
        plt.figure(figsize=(8, 6))
        sns.set_palette("husl")
        sns.scatterplot(data=clean_df, x="price", y="year")
        plt.xlabel("Mileage")
        plt.ticklabel_format(style="plain", axis="x")
        plt.ylabel("Year")
        plt.title("Relationship between price and year")
        st.pyplot(plt.gcf())

    with tab4:
        # Scatter plot of Price vs Mileage, with color indicating Status
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=clean_df, x="price", y="mileage", hue="status", palette="coolwarm"
        )
        plt.xlabel("Price")
        plt.ylabel("Mileage")
        plt.title("Relationship between Price, Mileage and Status")
        plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper right")
        plt.ticklabel_format(style="plain", axis="x")
        st.pyplot(plt.gcf())

    with tab5:
        # Scatter plot of Price vs Mileage, with color indicating Brand
        top_15_brands = (
            clean_df["brand"].value_counts().nlargest(15).index.to_list()
        )  # sorted by default
        filtered_by_brands_df = clean_df[clean_df["brand"].isin(top_15_brands)]
        plt.figure(figsize=(8, 6))
        sns.scatterplot(
            data=filtered_by_brands_df,
            x="mileage",
            y="price",
            hue="brand",
            palette="coolwarm",
        )
        plt.ylabel("Price")
        plt.xlabel("Mileage")
        plt.title("Relationship between Price, Mileage and Brand")
        plt.legend(title="Status", bbox_to_anchor=(1, 1), loc="upper right")
        st.pyplot(plt.gcf())

# Exploratory Data Analysis Tab
elif current_tab == "Exploratory Data Analysis":
    st.title("Exploratory Data Analysis")

    # Clean the dataset.
    clean_df = clean(cars_list_df)

    # Count the number of cars by status.
    dealers_group_counts = clean_df["status"].value_counts()

    # Create a pie chart to show the distribution of car statuses.
    plt.figure(figsize=(8, 8))
    plt.pie(
        dealers_group_counts,
        labels=dealers_group_counts.index,
        autopct="%1.1f%%",
        startangle=140,
    )
    plt.title("Number of cars per status")
    st.pyplot(plt.gcf())

    # Create a bar chart to show the distribution of car brands.
    plt.figure(figsize=(12, 6))
    clean_df.brand.value_counts().plot(kind="bar")
    plt.title("Brands distribution")
    plt.xlabel("Brands")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Create a bar chart to show the distribution of car years.
    plt.figure(figsize=(12, 6))
    clean_df.year.value_counts().sort_index().plot(kind="bar")
    plt.title("Year distribution")
    plt.xlabel("Year")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Create a bar chart to show the mean price per car brand.
    mean_price_per_brand = clean_df.groupby("brand")["price"].mean()
    plt.figure(figsize=(12, 6))
    mean_price_per_brand.plot(kind="bar")
    plt.title("Year distribution")
    plt.xlabel("Year")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Filter the data to show only the top 15 car brands.
    top_15_brands = (
        clean_df["brand"].value_counts().nlargest(15).index.to_list()
    )  # sorted by default
    filtered_by_brands_df = clean_df[clean_df["brand"].isin(top_15_brands)]

    # Count the sales of each car model within the top brands.
    sales_count = (
        filtered_by_brands_df.groupby(["brand", "model"])
        .size()
        .reset_index(name="sales_count")
    )
    # Identify the most sold model for each brand.
    most_sold = (
        sales_count.groupby("brand").max().sort_values("sales_count", ascending=False)
    )
    most_sold = most_sold.reset_index()
    most_sold["brand_model"] = most_sold["brand"] + " " + most_sold["model"]

    # Create a bar chart to show the most sold models.
    plt.figure(figsize=(12, 6))
    sns.barplot(data=most_sold, x="brand_model", y="sales_count")
    plt.title("Quantity distribution over models")
    plt.xlabel("Model")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Identify the least sold model for each brand.
    least_sold = (
        sales_count.groupby("brand").min().sort_values("sales_count", ascending=False)
    )
    least_sold = least_sold.reset_index()
    least_sold["brand_model"] = least_sold["brand"] + " " + least_sold["model"]

    # Create a bar chart to show the least sold models.
    plt.figure(figsize=(12, 6))
    sns.barplot(data=least_sold, x="brand_model", y="sales_count")
    plt.title("Quantity distribution over models")
    plt.xlabel("Model")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Filter the data to show only the top 15 dealers.
    top_15_dealers = clean_df["dealer"].value_counts().nlargest(15).index.to_list()
    filtered_by_dealers_df = clean_df[clean_df["dealer"].isin(top_15_dealers)]

    # Count the sales of each brand within the top dealers.
    sales_count = (
        filtered_by_dealers_df.groupby(["dealer", "brand"])
        .size()
        .reset_index(name="sales_count")
    )

    # Identify the most sold brand by each dealer.
    most_sold_by_dealer = (
        sales_count.groupby("dealer").max().sort_values("sales_count", ascending=False)
    )
    most_sold_by_dealer = most_sold_by_dealer.reset_index()
    most_sold_by_dealer["dealer_brand"] = (
        most_sold_by_dealer["dealer"] + "-" + most_sold_by_dealer["brand"]
    )
    # Create a bar chart to show the most sold brands by dealer.
    plt.figure(figsize=(12, 6))
    sns.barplot(data=most_sold_by_dealer, x="dealer_brand", y="sales_count")
    plt.title("Quantity distribution over models")
    plt.xlabel("Model")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Count the sales of each model within the top dealers.
    sales_count = (
        filtered_by_dealers_df.groupby(["dealer", "brand", "model"])
        .size()
        .reset_index(name="sales_count")
    )

    # Identify the most sold model by each dealer.
    most_sold_model_by_dealer = (
        sales_count.groupby("dealer").max().sort_values("sales_count", ascending=False)
    )
    most_sold_model_by_dealer = most_sold_model_by_dealer.reset_index()
    most_sold_model_by_dealer["brand_model"] = (
        most_sold_model_by_dealer["brand"] + "-" + most_sold_model_by_dealer["model"]
    )

    # Create a bar chart to show the most sold models by dealer.
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=most_sold_model_by_dealer,
        x="dealer",
        y="sales_count",
        hue="brand",
        palette="coolwarm",
    )
    plt.title("Quantity distribution over models")
    plt.xlabel("Model")
    plt.ylabel("Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(plt.gcf())
