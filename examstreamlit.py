# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load data dynamically from GitHub
@st.cache
def load_data():
    url = 'clean_df (2).csv'
    df = pd.read_csv(url)
    return df

# Load data
df = load_data()

# Streamlit Title and Description
st.title("Automobile Data Analysis")
st.markdown("""
### Explore and Analyze Features That Affect Automobile Prices
Use the options below to dynamically interact with the dataset and visualizations.
""")

# Sidebar options
st.sidebar.header("Filter Options")
selected_column = st.sidebar.selectbox("Select a column to explore:", df.columns)

# Display dataset
st.header("Dataset Overview")
st.write(df.head())

# Show data types
if st.sidebar.checkbox("Show Data Types"):
    st.subheader("Data Types")
    st.write(df.dtypes)

# Feature Pattern Analysis
st.header("Feature Pattern Analysis")
if df[selected_column].dtype in ['int64', 'float64']:
    st.subheader(f"Scatter Plot: {selected_column} vs Price")
    sns.regplot(x=selected_column, y="price", data=df)
    st.pyplot(plt)

    correlation = df[[selected_column, 'price']].corr().iloc[0, 1]
    st.write(f"Correlation between {selected_column} and price: {correlation}")
else:
    st.write("Selected column is non-numeric. Scatter plots are not applicable.")

# Descriptive Statistics
if st.sidebar.checkbox("Show Descriptive Statistics"):
    st.header("Descriptive Statistics")
    st.write(df.describe())

# Grouping and Aggregation
if st.sidebar.checkbox("Show Grouped Analysis"):
    st.header("Grouped Analysis")
    group_column = st.selectbox("Select column to group by:", ['drive-wheels', 'body-style'])
    grouped_data = df.groupby(group_column)['price'].mean().reset_index()
    st.write(grouped_data)
    st.bar_chart(grouped_data.set_index(group_column))
