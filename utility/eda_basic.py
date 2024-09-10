class EDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDA class with a DataFrame.

        Parameters:
        df (pd.DataFrame): The DataFrame to analyze.
        """
        self.df = df

    def display_branch_data(self):
        """
        Displays the DataFrame, its description, and missing values percentage
        in a Streamlit application.
        """
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