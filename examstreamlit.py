import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# Load dataset
@st.cache_data
def load_data():
    url = 'https://raw.githubusercontent.com/klamsal/Fall2024Exam/refs/heads/main/CleanedAutomobile.csv'
    return pd.read_csv(url)

df = load_data()

# App Title
st.title("Car Price Analysis")
st.write("Explore relationships between car features and their impact on car prices.")

# Show Dataset
st.header("Dataset")
if st.checkbox("Show raw dataset"):
    st.dataframe(df)

# Data Types
st.header("Data Types")
if st.checkbox("Show data types"):
    st.write(df.dtypes)

# Correlation Analysis
st.header("Correlation Analysis")
selected_columns = st.multiselect(
    "Select columns to calculate correlation:",
    df.select_dtypes(include=["float64", "int64"]).columns
)
if selected_columns:
    st.write("Correlation Matrix:")
    st.write(df[selected_columns].corr())

# Scatterplot for Continuous Variables
st.header("Scatterplots")
scatter_var = st.selectbox(
    "Select a feature to plot against price:",
    df.select_dtypes(include=["float64", "int64"]).columns
)
if scatter_var:
    st.subheader(f"Scatterplot: {scatter_var} vs Price")
    fig, ax = plt.subplots()
    sns.regplot(x=scatter_var, y="price", data=df, ax=ax)
    st.pyplot(fig)

# Boxplot for Categorical Variables
st.header("Boxplots")
boxplot_var = st.selectbox(
    "Select a categorical feature to analyze price distribution:",
    df.select_dtypes(include=["object"]).columns
)
if boxplot_var:
    st.subheader(f"Boxplot: {boxplot_var} vs Price")
    fig, ax = plt.subplots()
    sns.boxplot(x=boxplot_var, y="price", data=df, ax=ax)
    st.pyplot(fig)

# Descriptive Statistics
st.header("Descriptive Statistics")
if st.checkbox("Show summary statistics for numerical columns"):
    st.write(df.describe())

# Value Counts
st.header("Value Counts")
value_count_column = st.selectbox(
    "Select a column to see value counts:",
    df.columns
)
if value_count_column:
    st.write(df[value_count_column].value_counts().to_frame())

# Grouping Data
st.header("Grouping Data")
group_var = st.selectbox(
    "Group data by:",
    ["drive-wheels", "body-style", "engine-location"]
)
if group_var:
    grouped_data = df.groupby(group_var)["price"].mean().reset_index()
    st.subheader(f"Average Price by {group_var}")
    st.write(grouped_data)

# Pivot Table Heatmap
st.header("Heatmap: Drive-Wheels and Body-Style vs Price")
if st.checkbox("Show heatmap"):
    pivot_table = df.pivot_table(
        values="price", index="drive-wheels", columns="body-style", aggfunc="mean", fill_value=0
    )
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdBu", ax=ax)
    st.pyplot(fig)

# Correlation and P-Values
st.header("Pearson Correlation")
corr_var = st.selectbox(
    "Select a feature to calculate correlation and P-value with price:",
    df.select_dtypes(include=["float64", "int64"]).columns
)
if corr_var:
    pearson_coef, p_value = stats.pearsonr(df[corr_var], df["price"])
    st.write(f"**Pearson Correlation Coefficient:** {pearson_coef:.3f}")
    st.write(f"**P-value:** {p_value:.3f}")
    if p_value < 0.05:
        st.success("The correlation is statistically significant.")
    else:
        st.warning("The correlation is not statistically significant.")

st.write("Explore further with the provided dropdowns and checkboxes.")
