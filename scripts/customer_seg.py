import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class User_Overview:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def User_seg(self):
        # Calculate total session duration and total data (DL + UL) per user
        # Drop rows with missing values
        self.df.dropna(subset=['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)'], inplace=True)

        user_data = self.df.groupby('IMSI').agg(
            total_duration=('Dur. (ms)', 'sum'),
            total_DL=('Total DL (Bytes)', 'sum'),
            total_UL=('Total UL (Bytes)', 'sum')
        ).reset_index()

        # Calculate total data (DL + UL) for each user
        user_data['Total Data (Bytes)'] = user_data['total_DL'] + user_data['total_UL']

        # Segment users into deciles based on total session duration
        user_data['Decile'] = pd.qcut(user_data['total_duration'], 10, labels=False)

        return user_data
    def plot_insights(self):
        user_data = self.User_seg()
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Decile', y='Total Data (Bytes)', data=user_data, color='skyblue')
        plt.title('Total Data Usage by Decile of Users')
        plt.xlabel('Decile')
        plt.ylabel('Total Data (Bytes)')
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
    def calculate_engagement_metrics(self):
        # Calculate session frequency per user
        session_freq = self.df.groupby('IMSI').size().reset_index(name='Session Frequency')
        
        # Calculate total session duration per user
        total_duration = self.df.groupby('IMSI')['Dur. (ms)'].sum().reset_index(name='Total Duration (ms)')
        
        # Calculate total data (DL + UL) per user
        total_traffic = self.df.groupby('IMSI').agg(
            total_DL=('Total DL (Bytes)', 'sum'),
            total_UL=('Total UL (Bytes)', 'sum')
        ).reset_index()
        
        total_traffic['Total Traffic (Bytes)'] = total_traffic['total_DL'] + total_traffic['total_UL']
        
        # Merge all engagement metrics into one DataFrame
        engagement_metrics = session_freq.merge(total_duration, on='IMSI').merge(total_traffic[['IMSI', 'Total Traffic (Bytes)']], on='IMSI')
        
        return engagement_metrics










