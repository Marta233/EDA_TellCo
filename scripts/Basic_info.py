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

    # def correlation_heatmap_num_att(self):
    #     num_col = self.df.select_dtypes(include=['float64', 'int64']).columns
    #     corr_matric = num_col.corr()
    #     plt.figure(figsize=(10,8))
    #     plt.title('correlation')
    #     sns.heatmap(corr_matric, annot = True)
    #     plt.show()