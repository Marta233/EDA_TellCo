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
        self.engagement_metrics = session_freq.merge(total_duration, on='IMSI').merge(total_traffic[['IMSI', 'Total Traffic (Bytes)']], on='IMSI')
        
        return self.engagement_metrics
    
    def aggregate_metrics(self):
        # Group by MSISDN and aggregate metrics
        aggregated_data = self.df.groupby('MSISDN/Number').agg(
            sessions_frequency=('Bearer Id', 'nunique'),
            total_duration=('Dur. (ms)', 'sum'),
            total_traffic_DL=('Total DL (Bytes)', 'sum'),
            total_traffic_UL=('Total UL (Bytes)', 'sum')
        ).reset_index()

        # Calculate total traffic
        aggregated_data['total_traffic'] = aggregated_data['total_traffic_DL'] + aggregated_data['total_traffic_UL']

        # Get top 10 customers by each engagement metric
        top_10_sessions = aggregated_data.nlargest(10, 'sessions_frequency')
        top_10_duration = aggregated_data.nlargest(10, 'total_duration')
        top_10_traffic = aggregated_data.nlargest(10, 'total_traffic')

        return top_10_sessions, top_10_duration, top_10_traffic
    
    def k_means_clustering(self):
        # Ensure engagement metrics are calculated
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        k = 3
        scaler = StandardScaler()
        metrics = ['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']
        scaled_data = scaler.fit_transform(self.engagement_metrics[metrics])

        # Run k-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        self.engagement_metrics['cluster'] = kmeans.fit_predict(scaled_data)

        # Plotting the clusters
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            x=self.engagement_metrics['Session Frequency'],
            y=self.engagement_metrics['Total Duration (ms)'],
            hue=self.engagement_metrics['cluster'],
            palette='viridis',
            style=self.engagement_metrics['cluster'],
            s=100
        )
        
        plt.title('K-Means Clustering of User Engagement Metrics')
        plt.xlabel('Session Frequency')
        plt.ylabel('Total Duration (ms)')
        plt.legend(title='Cluster')
        plt.grid(True)
        plt.show()

        return self.engagement_metrics, kmeans
    
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
        return cluster_stats
    
    def top_engaged_users_per_app(self):
        # Ensure engagement metrics are calculated
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        app_columns = ['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                    'YouTube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)']

        # Aggregate total traffic per user for each app
        aggregated_data = self.df.groupby('MSISDN').agg({col: 'sum' for col in app_columns}).reset_index()

        # Derive the top 10 most engaged users for each application
        top_10_users_per_app = {app: aggregated_data.nlargest(10, app) for app in app_columns}

        return top_10_users_per_app

    def plot_top_apps(self):
        # Ensure engagement metrics are calculated
        if self.engagement_metrics is None:
            self.calculate_engagement_metrics()
        
        top_apps = ['Social Media DL (Bytes)', 'Youtube DL (Bytes)', 'Netflix DL (Bytes)']
        plt.figure(figsize=(12, 8))
        
        for app in top_apps:
            sns.barplot(x='MSISDN/Number', y=app, data=self.df.nlargest(10, app))
            plt.title(f'Top 10 Users for {app}')
            plt.xlabel('User ID (MSISDN)')
            plt.ylabel('Total Data (Bytes)') 
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
