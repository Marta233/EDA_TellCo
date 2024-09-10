import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.cluster import KMeans
from scripts.Experience_user import User_Experience
from scripts.user_engagment import User_Engagement
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sqlalchemy import create_engine
import os

class SatisfactionAnalysis:
    def __init__(self, df):
        """Initialize with the user DataFrame"""
        self.df = df

    def calculate_scores(self):
        # Create instances of analyzers
        engagement_analyzer = User_Engagement(self.df)
        experience_analyzer = User_Experience(self.df)
        
        # Calculate engagement and experience metrics
        engagement_metrics = engagement_analyzer.k_means_clustering()
        experience_metrics = experience_analyzer.cluster_user_exp_metrics()
        
        # Ensure metrics are DataFrames
        if isinstance(engagement_metrics, tuple):
            engagement_metrics = engagement_metrics[0]
        
        if isinstance(experience_metrics, tuple):
            experience_metrics = experience_metrics[0]
        
        # Get centroids from the analyzers
        engagement_centroids = engagement_analyzer.engagement_centroids
        experience_centroids = experience_analyzer.experience_centroids

        # Check if engagement_metrics is a DataFrame
        if not isinstance(engagement_metrics, pd.DataFrame):
            raise TypeError("engagement_metrics should be a DataFrame.")
        
        # Calculate engagement scores
        least_engaged_cluster = engagement_metrics['cluster'].value_counts().idxmin()
        least_engaged_centroid = engagement_centroids[least_engaged_cluster]
        engagement_metrics['engagement_score'] = engagement_metrics.apply(
            lambda row: euclidean(row[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']], least_engaged_centroid),
            axis=1
        )
        
        # Calculate experience scores
        worst_experience_cluster = experience_metrics['cluster'].value_counts().idxmax()
        worst_experience_centroid = experience_centroids[worst_experience_cluster]
        experience_metrics['experience_score'] = experience_metrics.apply(
            lambda row: euclidean(row[['Average TCP Retransmission', 'Average RTT', 'Average Throughput']], worst_experience_centroid),
            axis=1
        )
        
        # Merge scores with user data
        final_scores = engagement_metrics.merge(experience_metrics[['MSISDN/Number', 'experience_score']], on='MSISDN/Number')
        final_scores['satisfaction_score'] = (final_scores['engagement_score'] + final_scores['experience_score']) / 2
        
        return final_scores

    def assign_engagement_scores(self):
        # Create instance of analyzer
        engagement_analyzer = User_Engagement(self.df)
        
        # Calculate engagement metrics
        engagement_metrics = engagement_analyzer.k_means_clustering()
        
        # Ensure engagement_metrics is a DataFrame
        if isinstance(engagement_metrics, tuple):
            engagement_metrics = engagement_metrics[0]
        
        # Get centroids from the analyzers
        engagement_centroids = engagement_analyzer.engagement_centroids
        
        # Calculate Euclidean distance for each user
        least_engaged_cluster = np.argmin([np.mean(engagement_centroids[i]) for i in range(len(engagement_centroids))])
        least_engaged_centroid = engagement_centroids[least_engaged_cluster]
        engagement_metrics['Engagement Score'] = engagement_metrics.apply(
            lambda row: euclidean(row[['Session Frequency', 'Total Duration (ms)', 'Total Traffic (Bytes)']], least_engaged_centroid),
            axis=1
        )
        
        return engagement_metrics

    def assign_experience_scores(self):
        # Create instance of analyzer
        experience_analyzer = User_Experience(self.df)
        
        # Calculate experience metrics
        experience_metrics = experience_analyzer.cluster_user_exp_metrics()
        
        # Ensure experience_metrics is a DataFrame
        if isinstance(experience_metrics, tuple):
            experience_metrics = experience_metrics[0]
        
        # Get centroids from the analyzers
        experience_centroids = experience_analyzer.experience_centroids
        
        # Calculate Euclidean distance for each user
        worst_experience_cluster = np.argmax([np.mean(experience_centroids[i]) for i in range(len(experience_centroids))])
        worst_experience_centroid = experience_centroids[worst_experience_cluster]
        experience_metrics['Experience Score'] = experience_metrics.apply(
            lambda row: euclidean(row[['Average TCP Retransmission', 'Average RTT', 'Average Throughput']], worst_experience_centroid),
            axis=1
        )
        
        return experience_metrics

    def calculate_satisfaction_scores(self):
        engagement_metrics = self.assign_engagement_scores()
        experience_metrics = self.assign_experience_scores()
        combined_metrics = engagement_metrics.merge(experience_metrics, on='MSISDN/Number', suffixes=('_engagement', '_experience'))
        combined_metrics['Satisfaction Score'] = (combined_metrics['Engagement Score'] + combined_metrics['Experience Score']) / 2
        
        return combined_metrics

    def top_10_satisfied(self):
        combined_metrics = self.calculate_satisfaction_scores()
        top_10_satisfied = combined_metrics.nlargest(10, 'Satisfaction Score')
        print(top_10_satisfied[['MSISDN/Number', 'Satisfaction Score']])

    def plot_top_10_satisfied(self):
        combined_metrics = self.calculate_satisfaction_scores()
        top_10_satisfied = combined_metrics.nlargest(10, 'Satisfaction Score')
        plt.figure(figsize=(10, 6))
        sns.barplot(y='Satisfaction Score', x='MSISDN/Number', data=top_10_satisfied)
        plt.title('Top 10 Satisfied Customers')
        plt.xlabel('MSISDN/Number')
        plt.ylabel('Satisfaction Score')
        plt.xticks(rotation=45, ha='right')
        plt.show()

    def build_regression_model(self):
        # Start an MLflow run
        with mlflow.start_run():
            # Track the start time
            start_time = datetime.now()

            data = self.calculate_satisfaction_scores()
            X = data[['Engagement Score', 'Experience Score']]
            y = data['Satisfaction Score']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Log parameters
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)

            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)

            # Predict and calculate metrics
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

            # Log metrics
            mlflow.log_metric("mean_squared_error", mse)

            # Log the model itself
            mlflow.sklearn.log_model(model, "model")

            # Log any other artifacts like plots (optional)
            # Example: mlflow.log_artifact("path_to_plot.png")

            # Track the end time
            end_time = datetime.now()

            # Log start and end time
            mlflow.log_param("start_time", start_time)
            mlflow.log_param("end_time", end_time)

            return model, mse

    def cluster_scores(self):
        data = self.calculate_satisfaction_scores()
        scores = data[['Engagement Score', 'Experience Score']]
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        data['Cluster'] = kmeans.fit_predict(scores)
        
        return data

    def aggregate_scores_by_cluster(self):
        data = self.cluster_scores()
        cluster_aggregation = data.groupby('Cluster').agg({
            'Satisfaction Score': 'mean',
            'Experience Score': 'mean'
        }).reset_index()
        
        return cluster_aggregation

    def export_to_postgresql(self, table_name):
        """Export DataFrame to PostgreSQL database"""
        db_name = os.getenv('DB_NAME')
        db_user = os.getenv('DB_USER')
        db_password = os.getenv('DB_PASSWORD')
        db_host = os.getenv('DB_HOST')
        db_port = os.getenv('DB_PORT')

        # Construct the database URL
        db_url = f'postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'

        data = self.cluster_scores()
        # Ensure correct column selection
        data = data[['MSISDN/Number', 'Engagement Score', 'Experience Score', 'Satisfaction Score']]
        engine = create_engine(db_url)
        data.to_sql(table_name, engine, if_exists='replace', index=False)
        print(f"Data exported to table '{table_name}' in PostgreSQL database.")
        return table_name
