import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def basic_info(self):
        # Show basic information about the data
        print(self.df.info())

    def summary_statistics(self):
        # Display summary statistics
        print(self.df.describe())

    def missing_values(self):
        # Show missing values
        print(self.df.isnull().sum())
    def top_ten_handset(self):
        top_count = self.df['Handset Type'].value_counts().head(10)
        return top_count
    def plot_bar(self):
        top_count = self.df['Handset Type'].value_counts().nlargest(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("Top Ten Handset_Type used")
        plt.xlabel('Handset_Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.show()
