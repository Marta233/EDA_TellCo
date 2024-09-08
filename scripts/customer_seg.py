import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
class User_Overview:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def remove_outliers(self):
        # Drop rows where 'IMSI' or 'Dur. (ms)' is NaN
        self.df.dropna(subset=['IMSI', 'Dur. (ms)'], inplace=True)

        # Extract the IMSI column for outlier identification based on another metric
        imsi_col = self.df['IMSI']
        duration_col = self.df['Dur. (ms)']

        # Calculate Q1, Q3, and IQR for 'Dur. (ms)' to find outliers
        Q1 = duration_col.quantile(0.25)
        Q3 = duration_col.quantile(0.75)
        IQR = Q3 - Q1

        # Define outlier bounds
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers in 'Dur. (ms)' column
        outlier_mask = (duration_col < lower_bound) | (duration_col > upper_bound)
        outlier_imsi = self.df[outlier_mask]['IMSI']

        # Remove outliers based on IMSI
        df_cleaned = self.df[~self.df['IMSI'].isin(outlier_imsi)]

        # Display the cleaned DataFrame
        return df_cleaned
    def segment_and_compute_data(self):
        # Drop rows with missing values
        self.df.dropna(subset=['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'], inplace=True)
        self.remove_outliers()  
        user_aggregated = self.df.groupby('IMSI').agg(
            total_duration=('Dur. (ms)', 'sum'),
            total_DL=('Total DL (Bytes)', 'sum'),
            total_UL=('Total UL (Bytes)', 'sum')
        ).reset_index()

        # Step 2: Calculate deciles
        user_aggregated['Decile'] = pd.qcut(user_aggregated['total_duration'], 10, labels=False) + 1

        # Step 3: Focus on the top 5 deciles
        top_five_deciles = user_aggregated[user_aggregated['Decile'] > 5]

        # Step 4: Compute total data per decile
        decile_data = top_five_deciles.groupby('Decile').agg(
            total_users=('IMSI', 'count'),
            total_DL=('total_DL', 'sum'),
            total_UL=('total_UL', 'sum')
        ).reset_index()

       # Plot the data
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Plot Total DL
        sns.barplot(x='Decile', y='total_DL', data=decile_data, color='skyblue', ax=axes[0])
        axes[0].set_xlabel('Decile')
        axes[0].set_ylabel('Total DL (Bytes)')
        axes[0].set_title('Total Data Usage (DL) by Decile (Top 5)', fontsize=14, loc='center')
        
        # Plot Total UL
        sns.barplot(x='Decile', y='total_UL', data=decile_data, color='salmon', ax=axes[1])
        axes[1].set_xlabel('Decile')
        axes[1].set_ylabel('Total UL (Bytes)')
        axes[1].set_title('Total Data Usage (UL) by Decile (Top 5)', fontsize=14, loc='center')

        plt.tight_layout()
        plt.show()
    def non_graphical_univariate_analysis(self):
        # List of quantitative columns to analyze
        quantitative_cols = ['Dur. (ms)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
                            'Email DL (Bytes)', 'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
        
        for col in quantitative_cols:
            print(f"Analysis for {col}:")
            print(f"Mean: {self.df[col].mean()}")
            print(f"Median: {self.df[col].median()}")
            print(f"Mode: {self.df[col].mode()[0]}")
            print(f"Standard Deviation: {self.df[col].std()}")
            print(f"Variance: {self.df[col].var()}")
            print(f"Interquartile Range: {self.df[col].quantile(0.75) - self.df[col].quantile(0.25)}")
            print(f"Min: {self.df[col].min()}")
            print(f"Max: {self.df[col].max()}")
            print(f"Missing values: {self.df[col].isnull().sum()} / {len(self.df)}")
            print("="*50)
    def graphical_univariate_analysis(self):
        quantitative_cols = ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)', 'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
                            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
                            'Email DL (Bytes)', 'Email UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)', 'Other DL (Bytes)', 'Other UL (Bytes)']
        
        for col in quantitative_cols:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

            plt.figure(figsize=(10, 6))
            sns.boxplot(x=self.df[col].dropna())
            plt.title(f'Box Plot of {col}')
            plt.show()
    def bivariate_analysis(self):
        applications = ['Social Media DL (Bytes)', 'Netflix DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        self.df['Total Data (Bytes)'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']
        
        for app in applications:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=self.df[app], y=self.df['Total Data (Bytes)'])
            plt.title(f'Total Data Usage vs {app}')
            plt.show()
    def correlation_analysis(self):
        # Selecting relevant columns
        correlation_cols = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
        correlation_matrix = self.df[correlation_cols].corr()
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Matrix of Application Data Usage')
        plt.show()

    def dimensionality_reduction(self):
            # Select columns for PCA
            pca_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 'Other DL (Bytes)']
            # Standardize the data
            scaler = StandardScaler() 
            scaled_data = scaler.fit_transform(self.df[pca_columns].dropna())
            
            # Perform PCA
            pca = PCA(n_components=2)  # Reduce to 2 dimensions
            pca_result = pca.fit_transform(scaled_data)
            
            # Visualize the PCA result
            plt.figure(figsize=(10, 6))
            plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
            plt.title('PCA: Data in Reduced Dimensions')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
            plt.show()
            
            return pca.explained_variance_ratio_
   









