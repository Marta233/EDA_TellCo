import pandas as pd
import streamlit as st

pd.set_option("styler.render.max_elements", 8400056)

class EDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDA class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df

    def display_branch_data(self):
        if self.df.empty:
            st.warning("The DataFrame is empty. Please provide valid data.")
            return

        col1, col2, col3 = st.columns([1.5, 1, 0.75])
        
        with col1.expander("Data"):
            st.write(self.df.style.background_gradient(cmap="Purples"))
        
        with col2.expander("Description"):
            st.write(self.df.describe().T)
        
        with col3.expander("Missing Values"):
            st.subheader("Missing Values Percentage")
            missing_percent = self.df.isnull().sum() / len(self.df) * 100
            missing_df = pd.DataFrame({
                'Column': self.df.columns,
                'Missing Percentage': missing_percent
            }).sort_values(by='Missing Percentage', ascending=False)
            st.write(missing_df)

def show_customer_overview(data: pd.DataFrame):
    """
    Displays an overview of customer data using the EDA class.

    Parameters:
    data (pd.DataFrame): The DataFrame containing customer data.
    """
    st.subheader("Data Overview")
    eda = EDA(data)
    eda.display_branch_data()