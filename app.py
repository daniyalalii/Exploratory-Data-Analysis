# lab_eda_gui.py

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="EDA Dashboard", layout="wide", initial_sidebar_state="expanded"
)

st.title("Exploratory Data Analysis Interface")


# 2. Sidebar: Dataset Ingestion
# set header for sidebar
st.sidebar.header("DataSet Controls")

# create a file uploader in the sidebar for CSV files
uploaded_file = st.sidebar.file_uploader(
    "Please upload csv file for Analysis", type=["csv"]
)


if uploaded_file is not None:
    # Read dataset
    df = pd.read_csv(uploaded_file)

    # 3. Dataset Overview
    st.subheader("Dataset Preview & Metadata")

    # set subheader for dataset overview
    st.write("**First 5 Rows:**")
    # display the first 5 rows of the dataset
    st.dataframe(df.head())

    # display the shape of the dataset
    st.write(f"**Shape:** `{df.shape[0]}` rows, `{df.shape[1]}` columns")

    st.write("**Column Data Types:**")
    # display the data types of each column in the dataset
    dtypes_df = pd.DataFrame(df.dtypes, columns=["Data Type"])
    st.dataframe(dtypes_df)

    # Missing value summary
    st.write("**Missing Values per Column:**")
    # display the count and percentage of missing values for each column in the dataset
    missing_df = pd.DataFrame(
        {
            "Missing Count": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df)) * 100,
        }
    )
    st.dataframe(missing_df)

    # Basic statistics for numerical columns
    st.write("**Basic Numerical Statistics:**")
    # display the basic statistics (mean, median/50%, min, max)
    st.dataframe(
        df.describe().T[["mean", "50%", "min", "max"]].rename(
            columns={"50%": "median"}
        )
    )

    st.markdown("---")

    # 4. Attribute Selection
    # set header for attribute selection in the sidebar
    st.sidebar.header("Attribute Selection")

    # create a selectbox in the sidebar to choose an attribute for visualization
    selected_column = st.sidebar.selectbox(
        "Select Attribute for Visualization", options=df.columns
    )

    # Detect column type
    col_data = df[selected_column]
    if pd.api.types.is_numeric_dtype(col_data) and col_data.nunique() > 10:
        column_type = "Numerical"
    else:
        column_type = "Categorical"

    # 5. Visualization Rendering
    st.subheader("Visualization")

    fig, ax = plt.subplots(figsize=(8, 4))

    if column_type == "Numerical":
        # Histogram with seaborn
        sns.histplot(col_data.dropna(), kde=True, ax=ax, color="skyblue")
        ax.set_title(f"Histogram of {selected_column}", fontsize=14)
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    else:
        # Bar chart for categorical
        value_counts = col_data.value_counts(dropna=False)
        sns.barplot(
            x=value_counts.index.astype(str),
            y=value_counts.values,
            ax=ax,
            palette="Blues_d",
        )
        ax.set_title(
            f"Frequency Count of {selected_column}", fontsize=14
        )
        ax.set_xlabel(selected_column)
        ax.set_ylabel("Count")


        st.pyplot(fig)

else:
    st.info("Please upload a CSV file to start EDA.")