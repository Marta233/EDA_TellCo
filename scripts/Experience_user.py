import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
class User_Experience:
    numeric_cols = ['TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)', 
                    'Avg RTT DL (ms)', 'Avg RTT UL (ms)', 'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)']
    categorical_col = 'Handset Type'
    def __init__(self, df: pd.DataFrame):
        self.df = df
        
# Function to handle missing values
    def handle_missing_values(self):
        # For numerical columns, replace missing values with mean
        for col in self.numeric_cols:
            mean_value = self.df[col].mean()
            self.df[col].fillna(mean_value, inplace=True)
        
        # For categorical columns, replace missing values with mode
        mode_value = self.df[self.categorical_col].mode()[0]
        self.df[self.categorical_col].fillna(mode_value, inplace=True)

        return self.df

    def handle_outliers(self):
        for col in self.numeric_cols:
            if col in self.df.columns:
                # Calculate Q1 (25th percentile) and Q3 (75th percentile)
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Define outlier thresholds
                lower_limit = Q1 - 1.5 * IQR
                upper_limit = Q3 + 1.5 * IQR
                
                # Replace outliers with median (or mean) of the column
                median_value = self.df[col].median()
                self.df[col] = np.where((self.df[col] < lower_limit) | (self.df[col] > upper_limit), median_value, self.df[col])
        
        return self.df

        # Function to aggregate data per customer
    def aggregate_per_customer(self):
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()

        # Compute average TCP retransmission (both DL and UL)
        self.df['Average TCP Retransmission'] = (self.df['TCP DL Retrans. Vol (Bytes)'] + self.df['TCP UL Retrans. Vol (Bytes)']) / 2

        # Compute average RTT (both DL and UL)
        self.df['Average RTT'] = (self.df['Avg RTT DL (ms)'] + self.df['Avg RTT UL (ms)']) / 2

        # Compute average throughput (both DL and UL)
        self.df['Average Throughput'] = (self.df['Avg Bearer TP DL (kbps)'] + self.df['Avg Bearer TP UL (kbps)']) / 2

        # Aggregate data per customer
        aggregated_df = self.df.groupby('MSISDN/Number').agg({
            'Average TCP Retransmission': 'mean',
            'Average RTT': 'mean',
            self.categorical_col: lambda x: x.mode()[0],  # Mode of handset type
            'Average Throughput': 'mean'
        }).reset_index()

        return aggregated_df
    # Function to plot the aggregated data
    def plot_aggregated_data(self):
        aggregated_df= self.aggregate_per_customer()
        plt.figure(figsize=(16, 12))

        # Plot for Average TCP Retransmission
        plt.subplot(2, 2, 1)
        sns.histplot(aggregated_df['Average TCP Retransmission'], kde=True, color='skyblue')
        plt.xlabel('Average TCP Retransmission (Bytes)')
        plt.title('Distribution of Average TCP Retransmission')

        # Plot for Average RTT
        plt.subplot(2, 2, 2)
        sns.histplot(aggregated_df['Average RTT'], kde=True, color='salmon')
        plt.xlabel('Average RTT (ms)')
        plt.title('Distribution of Average RTT')

        # Plot for Average Throughput
        plt.subplot(2, 2, 3)
        sns.histplot(aggregated_df['Average Throughput'], kde=True, color='lightgreen')
        plt.xlabel('Average Throughput (kbps)')
        plt.title('Distribution of Average Throughput')

        # Plot for Handset Type distribution
        plt.subplot(2, 2, 4)
        handset_counts = aggregated_df[self.categorical_col].value_counts()
        sns.barplot(x=handset_counts.index, y=handset_counts.values, palette='viridis')
        plt.xlabel('Handset Type')
        plt.ylabel('Number of Customers')
        plt.title('Distribution of Handset Types')

        plt.tight_layout()
        plt.show()
    def compute_values(self):
        aggregated_df = self.aggregate_per_customer()
        top_10_tcp = aggregated_df['Average TCP Retransmission'].nlargest(10)
        bottom_10_tcp = aggregated_df['Average TCP Retransmission'].nsmallest(10)
        most_frequent_tcp = aggregated_df['Average TCP Retransmission'].mode().values

        top_10_rtt = aggregated_df['Average RTT'].nlargest(10)
        bottom_10_rtt = aggregated_df['Average RTT'].nsmallest(10)
        most_frequent_rtt = aggregated_df['Average RTT'].mode().values

        top_10_throughput = aggregated_df['Average Throughput'].nlargest(10)
        bottom_10_throughput = aggregated_df['Average Throughput'].nsmallest(10)
        most_frequent_throughput = aggregated_df['Average Throughput'].mode().values

        return {
            'top_10_tcp': top_10_tcp,
            'bottom_10_tcp': bottom_10_tcp,
            'most_frequent_tcp': most_frequent_tcp,
            'top_10_rtt': top_10_rtt,
            'bottom_10_rtt': bottom_10_rtt,
            'most_frequent_rtt': most_frequent_rtt,
            'top_10_throughput': top_10_throughput,
            'bottom_10_throughput': bottom_10_throughput,
            'most_frequent_throughput': most_frequent_throughput
        }

    # Function to display results
    def display_results(self):
        results = self.compute_values()
        print("Top 10 TCP Retransmission Values:\n", results['top_10_tcp'])
        print("\nBottom 10 TCP Retransmission Values:\n", results['bottom_10_tcp'])
        print("\nMost Frequent TCP Retransmission Values:\n", results['most_frequent_tcp'])

        print("\nTop 10 RTT Values:\n", results['top_10_rtt'])
        print("\nBottom 10 RTT Values:\n", results['bottom_10_rtt'])
        print("\nMost Frequent RTT Values:\n", results['most_frequent_rtt'])

        print("\nTop 10 Throughput Values:\n", results['top_10_throughput'])
        print("\nBottom 10 Throughput Values:\n", results['bottom_10_throughput'])
        print("\nMost Frequent Throughput Values:\n", results['most_frequent_throughput'])
    def plot_top_bottom(self):
        results = self.compute_values()
        plt.figure(figsize=(18, 12))

        # Plot for Top 10 TCP Retransmission
        plt.subplot(3, 3, 1)
        sns.barplot(x=results['top_10_tcp'].index, y=results['top_10_tcp'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.title('Top 10 Customers by Average TCP Retransmission')

        # Plot for Bottom 10 TCP Retransmission
        plt.subplot(3, 3, 2)
        sns.barplot(x=results['bottom_10_tcp'].index, y=results['bottom_10_tcp'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.title('Bottom 10 Customers by Average TCP Retransmission')

        # Plot for Most Frequent TCP Retransmission
        plt.subplot(3, 3, 3)
        most_frequent_tcp = pd.Series(results['most_frequent_tcp'])
        sns.barplot(x=most_frequent_tcp.index, y=most_frequent_tcp.values, palette='viridis')
        plt.xlabel('Index')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.title('Most Frequent TCP Retransmission Values')

        # Plot for Top 10 RTT
        plt.subplot(3, 3, 4)
        sns.barplot(x=results['top_10_rtt'].index, y=results['top_10_rtt'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average RTT (ms)')
        plt.title('Top 10 Customers by Average RTT')

        # Plot for Bottom 10 RTT
        plt.subplot(3, 3, 5)
        sns.barplot(x=results['bottom_10_rtt'].index, y=results['bottom_10_rtt'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average RTT (ms)')
        plt.title('Bottom 10 Customers by Average RTT')

        # Plot for Most Frequent RTT
        plt.subplot(3, 3, 6)
        most_frequent_rtt = pd.Series(results['most_frequent_rtt'])
        sns.barplot(x=most_frequent_rtt.index, y=most_frequent_rtt.values, palette='viridis')
        plt.xlabel('Index')
        plt.ylabel('Average RTT (ms)')
        plt.title('Most Frequent RTT Values')

        # Plot for Top 10 Throughput
        plt.subplot(3, 3, 7)
        sns.barplot(x=results['top_10_throughput'].index, y=results['top_10_throughput'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average Throughput (kbps)')
        plt.title('Top 10 Customers by Average Throughput')

        # Plot for Bottom 10 Throughput
        plt.subplot(3, 3, 8)
        sns.barplot(x=results['bottom_10_throughput'].index, y=results['bottom_10_throughput'].values, palette='viridis')
        plt.xlabel('Customer Index')
        plt.ylabel('Average Throughput (kbps)')
        plt.title('Bottom 10 Customers by Average Throughput')

        # Plot for Most Frequent RTT
        plt.subplot(3, 3, 9)
        most_frequent_throughput = pd.Series(results['most_frequent_throughput'])
        sns.barplot(x=most_frequent_throughput.index, y=most_frequent_throughput.values, palette='viridis')
        plt.xlabel('Index')
        plt.ylabel('Average throughput (ms)')
        plt.title('Most Frequent throughput Values')

        plt.tight_layout()
        plt.show()
    def compute_avg_throughput_per_handset(self):
        # Group by handset type and calculate the average throughput
         # Compute average throughput (both DL and UL)
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        self.df['Total Throughput'] = self.df['Avg Bearer TP DL (kbps)'] + self.df['Avg Bearer TP UL (kbps)']

        avg_throughput = self.df.groupby('Handset Type')['Total Throughput'].mean().reset_index()
        avg_throughput.columns = ['Handset Type', 'avg_throughput']
        return avg_throughput

    def visualize_avg_throughput(self):
        avg_throughput = self.compute_avg_throughput_per_handset()
        # Visualize the distribution of average throughput per handset type
        # Sort by average throughput and take the top N
        avg_throughput = avg_throughput.sort_values(by='avg_throughput', ascending=False).head(10)
    
        plt.figure(figsize=(12, 8))
        sns.barplot(x='avg_throughput', y='Handset Type', data=avg_throughput, palette='viridis')
        plt.title('Top Handset Type per Average Throughput')
        plt.xlabel('Average Throughput')
        plt.ylabel('Handset Type')
        plt.tight_layout()
        plt.show()
    def compute_avg_tcp_retransmission_per_handset(self):
        self.df = self.handle_missing_values()
        self.df = self.handle_outliers()
        # Group by handset type and calculate the average TCP retransmission
         # Compute average TCP retransmission (both DL and UL)
        self.df['Total TCP Retransmission'] = self.df['TCP DL Retrans. Vol (Bytes)'] + self.df['TCP UL Retrans. Vol (Bytes)']
       
        avg_tcp_retransmission = self.df.groupby('Handset Type')['Total TCP Retransmission'].mean().reset_index()
        avg_tcp_retransmission.columns = ['Handset Type', 'avg_tcp_retransmission']
        return avg_tcp_retransmission

    def visualize_avg_tcp_retransmission(self):
        avg_tcp_retransmission = self.compute_avg_tcp_retransmission_per_handset()
        # Visualize the distribution of average TCP retransmission per handset type
        # Sort by average throughput and take the top N
        avg_tcp_retransmission = avg_tcp_retransmission.sort_values(by='avg_tcp_retransmission', ascending=False).head(10)
    
        plt.figure(figsize=(12, 8))
        sns.barplot(x='avg_tcp_retransmission', y='Handset Type', data=avg_tcp_retransmission, palette='magma')
        plt.title('Top Handset Type per Average TCP Retransmission ')
        plt.xlabel('Average TCP Retransmission')
        plt.ylabel('Handset Type')
        plt.tight_layout()
        plt.show()
