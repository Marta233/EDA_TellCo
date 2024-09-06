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
        pd.set_option('display.max_columns', None)  # Show all columns
        pd.set_option('display.width', None)  # Prevent line wrapping
        print(self.df.isnull().sum())
        # Reset display options to default if needed
        pd.reset_option('display.max_columns')
        pd.reset_option('display.width')
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
    def top_manufacturere_by_handsetnum(self):
       hand_grou = self.df.groupby(['Handset Type', 'Handset Manufacturer']).size().reset_index(name='Count')
       top_count = hand_grou.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
       return top_count
    def plot_top_manufacturere_by_handsetnum(self):
        hand_grou = self.df.groupby(['Handset Type', 'Handset Manufacturer']).size().reset_index(name='Count')
        top_count = hand_grou.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("top 3 Manufacturer by Handset Number")
        plt.xlabel('Handset Manufacturee')
        plt.ylabel('count')
    def top_ten_handset_by_manufacturer(self):
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        return top_count
    def plot_top_ten_handset_by_manufacturer(self):
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("top 3 Manufacturer by Handset Number")
        plt.xlabel('Handset Manufacturee')
        plt.ylabel('count')
    def top_5_handsets_per_top_3_handset_manufacturer(self):
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3).index
        top_5_handsets = self.df[self.df['Handset Manufacturer'].isin(top_count)].groupby(
            ['Handset Manufacturer', 'Handset Type']).size().groupby('Handset Manufacturer', group_keys=False).nlargest(
            5)
        return top_5_handsets
    def plot_top_5_handsets_per_top_3_handset_manufacturer(self):
        # Get the top 3 handset manufacturers based on the number of handsets
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3).index
        
        # Filter the DataFrame for these top 3 manufacturers
        filtered_df = self.df[self.df['Handset Manufacturer'].isin(top_count)]
        
        # Get the top 5 handsets for each of the top 3 manufacturers
        top_5_handsets = (filtered_df.groupby(['Handset Manufacturer', 'Handset Type'])
                        .size()
                        .groupby('Handset Manufacturer', group_keys=False)
                        .nlargest(5))
        
        # Prepare data for plotting
        plot_data = top_5_handsets.reset_index(name='count')
        
        # Plotting
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Handset Type', y='count', hue='Handset Manufacturer', data=plot_data)
        plt.title("Top 5 Handsets by Handset Manufacturer")
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Handset Manufacturer')
        plt.show()
    def aggregate_user_behavior(self):
        # Print existing columns for debugging
        print("Existing columns:", self.df.columns.tolist())
        
        required_columns = [
            'IMSI',  # or MSISDN/Number for user identification
            'Dur. (ms)',
            'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)'
        ]
        
        # # Check for missing columns
        # missing_cols = [col for col in required_columns if col not in self.df.columns]
        # if missing_cols:
        #     raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        
        # Aggregate data per user
        aggregated_data =self.df.groupby('IMSI').agg(
            number_of_xDR_sessions=('Bearer Id', 'nunique'),
            total_session_duration=('Dur. (ms)', 'sum'),
            total_DL_data=('Total DL (Bytes)', 'sum'),
            total_UL_data=('Total UL (Bytes)', 'sum'),
            total_HTTP_DL=('HTTP DL (Bytes)', 'sum'),
            total_HTTP_UL=('HTTP UL (Bytes)', 'sum'),
            total_Social_Media_DL=('Social Media DL (Bytes)', 'sum'),
            total_Social_Media_UL=('Social Media UL (Bytes)', 'sum'),
            total_Netflix_DL=('Netflix DL (Bytes)', 'sum'),
            total_Netflix_UL=('Netflix UL (Bytes)', 'sum'),
            total_Google_DL=('Google DL (Bytes)', 'sum'),
            total_Google_UL=('Google UL (Bytes)', 'sum'),
            total_Email_DL=('Email DL (Bytes)', 'sum'),
            total_Email_UL=('Email UL (Bytes)', 'sum'),
            total_Gaming_DL=('Gaming DL (Bytes)', 'sum'),
            total_Gaming_UL=('Gaming UL (Bytes)', 'sum'),
            Youtube_Gaming_DL = ('Youtube DL (Bytes)', 'sum'),
            Youtube_Gaming_UL =('Youtube UL (Bytes)', 'sum')
        ).reset_index()

        return aggregated_data
    
