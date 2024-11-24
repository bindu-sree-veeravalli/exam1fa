import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy import stats

# Set up the Streamlit app
st.title("Car Price Analysis and Prediction")
st.write("Explore the relationships between car features and price using visualizations and statistics.")

# Load the dataset
@st.cache_data
def load_data():
    url = "clean_df (2).csv"
    return pd.read_csv(url)

df = load_data()

# Display the dataset
st.header("Dataset")
st.dataframe(df)

# Data types
st.header("Column Data Types")
st.write(df.dtypes)

# Question 1: Data type of "peak-rpm"
st.subheader("Data Type of 'peak-rpm'")
st.write(f"'peak-rpm' is of type: {df['peak-rpm'].dtype}")

# Correlation analysis
st.header("Correlation Analysis")
columns_to_corr = ['bore', 'stroke', 'compression-ratio', 'horsepower']
correlation_matrix = df[columns_to_corr].corr()
st.subheader("Correlation Matrix for Selected Columns")
st.write(correlation_matrix)

# Scatterplots for Continuous Variables
st.header("Scatterplots")
scatter_var = st.selectbox("Choose a feature to plot against price:", df.select_dtypes(include=['float64', 'int64']).columns)

if scatter_var:
    fig, ax = plt.subplots()
    sns.regplot(x=scatter_var, y="price", data=df, ax=ax)
    ax.set_xlabel(scatter_var)
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Boxplots for Categorical Variables
st.header("Boxplots")
boxplot_var = st.selectbox("Choose a categorical variable to visualize price distribution:", df.select_dtypes(include=['object', 'category']).columns)

if boxplot_var:
    fig, ax = plt.subplots()
    sns.boxplot(x=boxplot_var, y="price", data=df, ax=ax)
    ax.set_xlabel(boxplot_var)
    ax.set_ylabel("Price")
    st.pyplot(fig)

# Descriptive Statistics
st.header("Descriptive Statistics")
if st.checkbox("Show descriptive statistics for numeric columns"):
    st.write(df.describe())

# Grouping and Pivot Tables
st.header("Grouping and Pivot Table")
grouped_option = st.selectbox("Group data by:", ["drive-wheels", "body-style", "engine-location"])
grouped_data = df.groupby([grouped_option], as_index=False)["price"].mean()
st.write(f"Average price grouped by {grouped_option}")
st.write(grouped_data)

# Correlation and Pearson Coefficients
st.header("Pearson Correlation")
pearson_feature = st.selectbox("Select a feature for correlation with price:", df.select_dtypes(include=['float64', 'int64']).columns)

if pearson_feature:
    pearson_coef, p_value = stats.pearsonr(df[pearson_feature], df['price'])
    st.write(f"Pearson Correlation Coefficient: {pearson_coef:.3f}")
    st.write(f"P-value: {p_value:.3f}")
    if p_value < 0.05:
        st.success("The correlation is statistically significant.")
    else:
        st.warning("The correlation is not statistically significant.")

# Heatmap for Pivot Table
st.header("Heatmap Visualization")
if st.checkbox("Show heatmap for drive-wheels and body-style vs price"):
    pivot_table = df.pivot_table(values='price', index='drive-wheels', columns='body-style', aggfunc='mean', fill_value=0)
    fig, ax = plt.subplots()
    sns.heatmap(pivot_table, annot=True, fmt=".0f", cmap="RdBu", ax=ax)
    st.pyplot(fig)

st.write("This app allows users to explore relationships between car features and prices. Use the dropdowns and checkboxes above to interact with the data.")
