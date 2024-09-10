import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

class User_Engagement:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.engagement_metrics = None
    def remove_outliers(self):
        # Drop rows where 'IMSI', 'MSISDN/Number', or 'Bearer Id' is NaN
        self.df.dropna(subset=['IMSI', 'MSISDN/Number', 'Bearer Id'], inplace=True)
        mean_value = self.df['Total DL (Bytes)'].mean()
        self.df['Total DL (Bytes)'].fillna(mean_value, inplace=True) 
        mean_value1 = self.df['Total UL (Bytes)'].mean()
        self.df['Total UL (Bytes)'].fillna(mean_value1, inplace=True) 
        # First attribute: 'MSISDN/Number' (based on frequency)
        msisdn_freq = self.df['MSISDN/Number'].value_counts()
        Q1_msisdn = msisdn_freq.quantile(0.25)
        Q3_msisdn = msisdn_freq.quantile(0.75)
        IQR_msisdn = Q3_msisdn - Q1_msisdn
        lower_bound_msisdn = Q1_msisdn - 1.5 * IQR_msisdn
        upper_bound_msisdn = Q3_msisdn + 1.5 * IQR_msisdn

        # Second attribute: 'Bearer Id' (based on frequency)
        bearer_freq = self.df['Bearer Id'].value_counts()
        Q1_bearer = bearer_freq.quantile(0.25)
        Q3_bearer = bearer_freq.quantile(0.75)
        IQR_bearer = Q3_bearer - Q1_bearer
        lower_bound_bearer = Q1_bearer - 1.5 * IQR_bearer
        upper_bound_bearer = Q3_bearer + 1.5 * IQR_bearer

        # Identify outliers in both 'MSISDN/Number' and 'Bearer Id' based on frequency
        msisdn_outliers = msisdn_freq[(msisdn_freq < lower_bound_msisdn) | (msisdn_freq > upper_bound_msisdn)].index
        bearer_outliers = bearer_freq[(bearer_freq < lower_bound_bearer) | (bearer_freq > upper_bound_bearer)].index

        # Remove outliers from DataFrame
        df_cleaned = self.df[~self.df['MSISDN/Number'].isin(msisdn_outliers) & ~self.df['Bearer Id'].isin(bearer_outliers)]

        
    def replace_outliers_with_mean(self):
        for column in ['Total UL (Bytes)', 'Total DL (Bytes)']:
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            mean_value = self.df[column].mean()
            self.df[column] = self.df[column].apply(lambda x: mean_value if x < lower_bound or x > upper_bound else x)

    def calculate_engagement_metrics(self):
        # Remove outliers and drop rows with missing values in key columns
        self.remove_outliers()
        self.replace_outliers_with_mean()
        
        # Calculate session frequency per user
        session_freq = self.df.groupby('MSISDN/Number').size().reset_index(name='Session Frequency')
        
        # Calculate total session duration per user
        total_duration = self.df.groupby('MSISDN/Number')['Dur. (ms)'].sum().reset_index(name='Total Duration (ms)')
        
        # Calculate total data usage per user (DL + UL)
        total_traffic = self.df.groupby('MSISDN/Number').agg(
            sessions_frequency=('Bearer Id', 'nunique'),
            total_DL=('Total DL (Bytes)', 'sum'),
            total_UL=('Total UL (Bytes)', 'sum')
        ).reset_index()
        
        total_traffic['Total Traffic (Bytes)'] = total_traffic['total_DL'] + total_traffic['total_UL']
        
        # Merge metrics into one DataFrame
        self.engagement_metrics = session_freq.merge(total_duration, on='MSISDN/Number').merge(total_traffic[['MSISDN/Number', 'Total Traffic (Bytes)']], on='MSISDN/Number')
        
        return self.engagement_metrics

    def plot_aggre_matrix(self):
        # Calculate the correlation matrix
        correlation_matrix = self.engagement_metrics.corr()

        # Create a heatmap of the correlation matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.show()
    def report_top_10_customers(self):
        # Calculate engagement metrics if not already done
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        # Report the top 10 customers per engagement metric
        top_10_session_freq = self.engagement_metrics.nlargest(10, 'Session Frequency')
        top_10_duration = self.engagement_metrics.nlargest(10, 'Total Duration (ms)')
        top_10_traffic = self.engagement_metrics.nlargest(10, 'Total Traffic (Bytes)')
        
        print("Top 10 Customers by Session Frequency:\n", top_10_session_freq[['MSISDN/Number', 'Session Frequency']])
        print("\nTop 10 Customers by Total Duration:\n", top_10_duration[['MSISDN/Number', 'Total Duration (ms)']])
        print("\nTop 10 Customers by Total Traffic:\n", top_10_traffic[['MSISDN/Number', 'Total Traffic (Bytes)']])

        return top_10_session_freq, top_10_duration, top_10_traffic
    def plot_top_10_customers(self):
        # Calculate engagement metrics if not already done
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()

        # Plot the top 10 customers based on session frequency
        top_10_session_freq = self.engagement_metrics.nlargest(10, 'Session Frequency')
        plt.figure(figsize=(10, 6))
        sns.barplot(y='Session Frequency', x='MSISDN/Number', data=top_10_session_freq)
        plt.title('Top 10 Customers by Session Frequency')
        plt.xlabel('MSISDN/Number')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Session Frequency')
        plt.show()

        top_10_duration = self.engagement_metrics.nlargest(10, 'Total Duration (ms)')
        plt.figure(figsize=(10, 6))
        sns.barplot(y='Total Duration (ms)', x='MSISDN/Number', data=top_10_duration)
        plt.title('Top 10 Customers by Total Duration')
        plt.xlabel('MSISDN/Number')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Total Duration (ms)')
        plt.show()

        top_10_traffic = self.engagement_metrics.nlargest(10, 'Total Traffic (Bytes)')
        plt.figure(figsize=(10, 6))
        sns.barplot(y='Total Traffic (Bytes)', x='MSISDN/Number', data=top_10_traffic)
        plt.title('Top 10 Customers by Total Traffic')
        plt.xlabel('MSISDN/Number')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Total Traffic (Bytes)')
        plt.show()
        
    
    def k_means_clustering(self):     
     # Get aggregated data
        engagement_metrics = self.calculate_engagement_metrics()

        # Ensure the required columns are present
        required_columns = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
        if not all(col in engagement_metrics.columns for col in required_columns):
            raise ValueError("Some required columns are missing from the DataFrame")

        k = 3
        scaler = StandardScaler()

        # Standardize the numerical features
        numerical_features = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
        scaled_numerical_data = scaler.fit_transform(engagement_metrics[numerical_features])

        # Run k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        engagement_metrics['cluster'] = kmeans.fit_predict(scaled_numerical_data)
        # Store the centroids for future use
        self.engagement_centroids = kmeans.cluster_centers_
        
        self.engagement_metrics = engagement_metrics  # Ensure this is assigned

        return self.engagement_metrics
    def plot_cluster(self):     
        # Call k_means_clustering to get the data with clusters
        df_clustering, _ = self.k_means_clustering()  # Unpack the tuple
        
        # Plotting the clusters     
        plt.figure(figsize=(12, 8))     
        
        sns.scatterplot(         
            x=df_clustering['Session Frequency'],         
            y=df_clustering['Total Duration (ms)'],         
            hue=df_clustering['cluster'],         
            palette='viridis',         
            style=df_clustering['cluster'],         
            s=100     
        )              
        
        plt.title('K-Means Clustering of User Engagement Metrics')     
        plt.xlabel('Session Frequency')     
        plt.ylabel('Total Duration (ms)')     
        plt.legend(title='Cluster')     
        plt.grid(True)     
        plt.show()



    
    def compute_cluster_stats(self):
        # Ensure engagement metrics are calculated
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        cluster_stats = self.engagement_metrics.groupby('cluster').agg(
            min_sessions_frequency=('Session Frequency', 'min'),
            max_sessions_frequency=('Session Frequency', 'max'),
            avg_sessions_frequency=('Session Frequency', 'mean'),
            total_sessions_frequency=('Session Frequency', 'sum'),
            min_duration=('Total Duration (ms)', 'min'),
            max_duration=('Total Duration (ms)', 'max'),
            avg_duration=('Total Duration (ms)', 'mean'),
            total_duration=('Total Duration (ms)', 'sum'),
            min_traffic=('Total Traffic (Bytes)', 'min'),
            max_traffic=('Total Traffic (Bytes)', 'max'),
            avg_traffic=('Total Traffic (Bytes)', 'mean'),
            total_traffic=('Total Traffic (Bytes)', 'sum')
        )
            # Plotting the cluster statistics
        cluster_stats_plot_data = cluster_stats[['avg_sessions_frequency', 'avg_duration', 'avg_traffic']]
        cluster_stats_plot_data.plot(kind='bar', figsize=(10, 6))

        plt.title('Average Session Frequency, Duration, and Traffic per Cluster')
        plt.xlabel('Cluster')
        plt.ylabel('Average Values')
        plt.xticks(rotation=0, ha = 'right')
        plt.tight_layout()
        
        plt.show()
    
        return cluster_stats
    def aggregate_traffic_per_application(self):
            # Aggregate total traffic (DL + UL) per user per application
            applications = ['Youtube DL (Bytes)', 'Social Media DL (Bytes)', 'Netflix DL (Bytes)', 
                            'Google DL (Bytes)', 'Gaming DL (Bytes)', 'Email DL (Bytes)', 
                            'Youtube UL (Bytes)', 'Social Media UL (Bytes)', 'Netflix UL (Bytes)', 
                            'Google UL (Bytes)', 'Gaming UL (Bytes)', 'Email UL (Bytes)']
            
            # Summing DL and UL for each application
            self.df['Total Youtube Traffic'] = self.df['Youtube DL (Bytes)'] + self.df['Youtube UL (Bytes)']
            self.df['Total Social Media Traffic'] = self.df['Social Media DL (Bytes)'] + self.df['Social Media UL (Bytes)']
            self.df['Total Netflix Traffic'] = self.df['Netflix DL (Bytes)'] + self.df['Netflix UL (Bytes)']
            self.df['Total Google Traffic'] = self.df['Google DL (Bytes)'] + self.df['Google UL (Bytes)']
            self.df['Total Gaming Traffic'] = self.df['Gaming DL (Bytes)'] + self.df['Gaming UL (Bytes)']
            self.df['Total Email Traffic'] = self.df['Email DL (Bytes)'] + self.df['Email UL (Bytes)']
            
            application_cols = ['Total Youtube Traffic', 'Total Social Media Traffic', 'Total Netflix Traffic', 
                                'Total Google Traffic', 'Total Gaming Traffic', 'Total Email Traffic']
            
            # Aggregate total traffic per application
            self.application_traffic = self.df.groupby('MSISDN/Number')[application_cols].sum().reset_index()

            return self.application_traffic

    def top_10_users_per_application(self):
        # Aggregate total traffic per application if not done already
        if self.application_traffic is None:
            self.aggregate_traffic_per_application()

        # Top 10 users per application based on total traffic
        top_10_youtube = self.application_traffic.nlargest(10, 'Total Youtube Traffic')
        top_10_social_media = self.application_traffic.nlargest(10, 'Total Social Media Traffic')
        top_10_netflix = self.application_traffic.nlargest(10, 'Total Netflix Traffic')

        top_10_google = self.application_traffic.nlargest(10, 'Total Google Traffic')
        top_10_gaminf = self.application_traffic.nlargest(10, 'Total Gaming Traffic')
        top_10_email = self.application_traffic.nlargest(10, 'Total Email Traffic')

        print("Top 10 Most Engaged Users on YouTube:\n", top_10_youtube[['MSISDN/Number', 'Total Youtube Traffic']])
        print("\nTop 10 Most Engaged Users on Social Media:\n", top_10_social_media[['MSISDN/Number', 'Total Social Media Traffic']])
        print("\nTop 10 Most Engaged Users on Netflix:\n", top_10_netflix[['MSISDN/Number', 'Total Netflix Traffic']])

        print("Top 10 Most Engaged Users on google:\n", top_10_google[['MSISDN/Number', 'Total Google Traffic']])
        print("\nTop 10 Most Engaged Users on gaming:\n", top_10_gaminf[['MSISDN/Number', 'Total Gaming Traffic']])
        print("\nTop 10 Most Engaged Users on email:\n", top_10_email[['MSISDN/Number', 'Total Email Traffic']])

        # Plotting
        fig, axs = plt.subplots(6, 1, figsize=(10, 15), sharex=True)

        # YouTube
        axs[0].barh(top_10_youtube['MSISDN/Number'].astype(str), top_10_youtube['Total Youtube Traffic'], color='blue')
        axs[0].set_title('Top 10 Users by YouTube Traffic')
        axs[0].set_xlabel('Total YouTube Traffic (Bytes)')
        axs[0].set_ylabel('MSISDN/Number')

        # Social Media
        axs[1].barh(top_10_social_media['MSISDN/Number'].astype(str), top_10_social_media['Total Social Media Traffic'], color='green')
        axs[1].set_title('Top 10 Users by Social Media Traffic')
        axs[1].set_xlabel('Total Social Media Traffic (Bytes)')
        axs[1].set_ylabel('MSISDN/Number')

        # Netflix
        axs[2].barh(top_10_netflix['MSISDN/Number'].astype(str), top_10_netflix['Total Netflix Traffic'], color='red')
        axs[2].set_title('Top 10 Users by Netflix Traffic')
        axs[2].set_xlabel('Total Netflix Traffic (Bytes)')
        axs[2].set_ylabel('MSISDN/Number')

         # google
        axs[3].barh(top_10_netflix['MSISDN/Number'].astype(str), top_10_netflix['Total Google Traffic'], color='red')
        axs[3].set_title('Top 10 Users by Google Traffic')
        axs[3].set_xlabel('Total Google Traffic (Bytes)')
        axs[3].set_ylabel('MSISDN/Number')
        # gaming
        axs[4].barh(top_10_netflix['MSISDN/Number'].astype(str), top_10_netflix['Total Gaming Traffic'], color='red')
        axs[4].set_title('Top 10 Users by gaming Traffic')
        axs[4].set_xlabel('Total gaming Traffic (Bytes)')
        axs[4].set_ylabel('MSISDN/Number')
        # Email
        axs[5].barh(top_10_netflix['MSISDN/Number'].astype(str), top_10_netflix['Total Email Traffic'], color='red')
        axs[5].set_title('Top 10 Users by Email Traffic')
        axs[5].set_xlabel('Total Email Traffic (Bytes)')
        axs[5].set_ylabel('MSISDN/Number')
       

        plt.tight_layout()
        plt.show()

        return top_10_youtube, top_10_social_media, top_10_netflix

    def plot_top_3_applications(self):
        # Aggregate total traffic per application if not done already
        if self.application_traffic is None:
            self.aggregate_traffic_per_application()

        # Calculate total traffic for each application
        total_traffic = self.application_traffic[['Total Youtube Traffic', 'Total Social Media Traffic', 
                                                'Total Netflix Traffic', 'Total Google Traffic', 
                                                'Total Gaming Traffic', 'Total Email Traffic']].sum()
        
        # Identify the top 3 most used applications
        top_3_applications = total_traffic.nlargest(3)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        
        plt.bar(top_3_applications.index, top_3_applications.values, color=['skyblue', 'lightgreen', 'coral'])
        plt.xlabel('Application')
        plt.ylabel('Total Traffic (Bytes)')
        plt.title('Top 3 Most Used Applications')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.show()

    def elbow_method(self):
        # Ensure engagement metrics are calculated
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        sse = []
        k_range = range(1, 11)
        scaler = StandardScaler()
        metrics = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
        scaled_data = scaler.fit_transform(self.engagement_metrics[metrics])
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(scaled_data)
            sse.append(kmeans.inertia_)

        plt.figure(figsize=(8, 5))
        plt.plot(k_range, sse, marker='o')
        plt.title('Elbow Method for Optimal k')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('SSE')
        plt.show()
