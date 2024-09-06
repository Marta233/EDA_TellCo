import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
class EDA:
    def __init__(self, df: pd.DataFrame):
        self.df = df
    def remove_columns(self):
        # List of columns to be dropped, removing duplicates
        columns_to_drop = list(set([
            'Nb of sec with 37500B < Vol UL',
       'Nb of sec with 6250B < Vol UL < 37500B', 'Nb of sec with 125000B < Vol DL',
       'TCP UL Retrans. Vol (Bytes)', 'Nb of sec with 31250B < Vol DL < 125000B',
       'Nb of sec with 1250B < Vol UL < 6250B', 'Nb of sec with 6250B < Vol DL < 31250B',
       'TCP DL Retrans. Vol (Bytes)', 'HTTP UL (Bytes)',
       'HTTP DL (Bytes)', 'Avg RTT DL (ms)',
       'Avg RTT UL (ms)', 'Last Location Name','Nb of sec with 125000B < Vol DL',
        ]))
        
        # Check which columns are actually in the DataFrame
        existing_columns = self.df.columns
        missing_columns = [col for col in columns_to_drop if col not in existing_columns]
        
        if missing_columns:
            print(f"Warning: The following columns were not found and will not be dropped: {missing_columns}")
        
        # Drop columns if they exist
        self.df = self.df.drop(columns=[col for col in columns_to_drop if col in existing_columns])
        
        return self.df

    def basic_info(self):
        # Set display options to show more columns
        pd.set_option('display.max_columns', None)  # None means no limit
        # Show basic information about the data
        print(self.df.info())
    def basic_info1(self):
        # Show basic information about the data
        self.remove_columns()
        self.df.info()
    def summary_statistics(self):
        # Display summary statistics
        print(self.df.describe())
    def missing_values(self):
        print(self.df.isnull().sum())
    def missing_percentage(self):
    # Calculate the percentage of missing values
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        
        # Create a DataFrame to display the results nicely
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        
        return missing_df
    def remove_null_spe_col(self):
        self.df = self.df.dropna(subset=['Handset Type', 'Handset Manufacturer'])
    def data_types(self):
        print(self.df.dtypes)
    def top_ten_handset(self):
        self.remove_null_spe_col()
        top_count = self.df['Handset Type'].value_counts().head(10)
        return top_count
    def plot_bar(self):
        self.remove_null_spe_col()
        top_count = self.df['Handset Type'].value_counts().nlargest(10)
        plt.figure(figsize=(10,6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("Top Ten Handset_Type used")
        plt.xlabel('Handset_Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.show()
    def top_manufacturere_by_handsetnum(self):
       self.remove_null_spe_col()
       hand_grou = self.df.groupby(['Handset Type', 'Handset Manufacturer']).size().reset_index(name='Count')
       top_count = hand_grou.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
       return top_count
    def plot_top_manufacturere_by_handsetnum(self):
        self.remove_null_spe_col()
        hand_grou = self.df.groupby(['Handset Type', 'Handset Manufacturer']).size().reset_index(name='Count')
        top_count = hand_grou.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("top 3 Manufacturer by Handset Number")
        plt.xlabel('Handset Manufacturee')
        plt.ylabel('count')
    def top_ten_handset_by_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        return top_count
    def plot_top_ten_handset_by_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("top 3 Manufacturer by Handset Number")
        plt.xlabel('Handset Manufacturee')
        plt.ylabel('count')
    def top_5_handsets_per_top_3_handset_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3).index
        top_5_handsets = self.df[self.df['Handset Manufacturer'].isin(top_count)].groupby(
            ['Handset Manufacturer', 'Handset Type']).size().groupby('Handset Manufacturer', group_keys=False).nlargest(
            5)
        return top_5_handsets
    def plot_top_5_handsets_per_top_3_handset_manufacturer(self):
        self.remove_null_spe_col()
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
            # 'HTTP DL (Bytes)', 'HTTP UL (Bytes)',
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
            # total_HTTP_DL=('HTTP DL (Bytes)', 'sum'),
            # total_HTTP_UL=('HTTP UL (Bytes)', 'sum'),
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
    def plot_insights(self):
        # Plot total data usage
        aggregated_data = self.aggregate_user_behavior()
        plt.figure(figsize=(12, 8))
        sns.histplot(aggregated_data['total_DL_data'], kde=True, color='blue')
        plt.title('Distribution of Total Download Data (Bytes)')
        plt.xlabel('Total Download Data (Bytes)')
        plt.ylabel('Frequency')
        plt.show()
    def plot_insight1(self):
        aggregated_data = self.aggregate_user_behavior()
        # Plot total session duration
        plt.figure(figsize=(12, 8))
        sns.histplot(aggregated_data['total_session_duration'], kde=True, color='green')
        plt.title('Distribution of Total Session Duration (ms)')
        plt.xlabel('Total Session Duration (ms)')
        plt.ylabel('Frequency')
        plt.show()
    def plot_insight3(self):
        # Plot top sources of data usage
        aggregated_data = self.aggregate_user_behavior()
        usage_columns = [
            'total_Social_Media_DL', 'total_Netflix_DL', 'total_Google_DL', 'total_Email_DL', 'total_Gaming_DL', 'Youtube_Gaming_DL'
        ]
        plt.figure(figsize=(14, 8))
        aggregated_data[usage_columns].sum().sort_values().plot(kind='barh', color='skyblue')
        plt.title('Total Data Usage by Source (Bytes)')
        plt.xlabel('Total Data Usage (Bytes)')
        plt.ylabel('Data Source')
        plt.show()
    def plot_insight4(self):
        aggregated_data = self.aggregate_user_behavior()
        # Plot number of sessions
        plt.figure(figsize=(12, 8))
        sns.histplot(aggregated_data['number_of_xDR_sessions'], kde=True, color='orange')
        plt.title('Distribution of Number of Sessions per User')
        plt.xlabel('Number of Sessions')
        plt.ylabel('Frequency')
        plt.show()
    def outlier_check(self):
        # Filter for numeric columns
        numeric_df = self.df.select_dtypes(include=[float, int])

        # Ensure there is at least one numeric column
        if numeric_df.empty:
            raise ValueError("No numeric columns available for outlier detection.")

        # Identify outliers using IQR method
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate outliers for each column
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        
        # Return the count of outliers for each column
        return outliers.sum()
    def outlier_check_perc(self):
        # Filter for numeric columns
        numeric_df = self.df.select_dtypes(include=[float, int])

        # Ensure there is at least one numeric column
        if numeric_df.empty:
            raise ValueError("No numeric columns available for outlier detection.")

        # Identify outliers using IQR method
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1

        # Calculate outliers for each column
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        
        # Count total values per column and calculate percentage of outliers
        total_values = numeric_df.count()  # Non-NaN values in each column
        outlier_percentage = (outliers.sum() / total_values) * 100
        
        # Create a DataFrame for outlier percentages
        missing_df = pd.DataFrame({
            'Column': numeric_df.columns,  # Use numeric_df.columns instead of self.df.columns
            'outlier_percentage': outlier_percentage
        }).sort_values(by='outlier_percentage', ascending=False)  # Fix typo
        
        return missing_df



