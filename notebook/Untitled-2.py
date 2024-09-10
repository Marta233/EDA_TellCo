def missing_percentage(self):
        st.subheader("Missing Values Percentage")
        missing_percent = self.df.isnull().sum() / len(self.df) * 100
        missing_df = pd.DataFrame({
            'Column': self.df.columns,
            'Missing Percentage': missing_percent
        }).sort_values(by='Missing Percentage', ascending=False)
        st.write(missing_df)

    def remove_null_spe_col(self):
        self.df = self.df.dropna(subset=['Handset Type', 'Handset Manufacturer'])

    def data_types(self):
        st.subheader("Data Types")
        data_types = self.df.dtypes
        data_types_df = pd.DataFrame({
            'Column': self.df.columns,
            'Data Type': data_types
        }).sort_values(by='Data Type', ascending=False)
        st.write(data_types_df)

    def top_ten_handset(self):
        self.remove_null_spe_col()
        top_count = self.df['Handset Type'].value_counts().head(10)
        return top_count

    def plot_bar(self):
        self.remove_null_spe_col()
        top_count = self.df['Handset Type'].value_counts().nlargest(10)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("Top Ten Handset Types Used")
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)
        plt.clf()  # Clear the figure

    def top_ten_handset_by_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        return top_count

    def plot_top_ten_handset_by_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=top_count.index, y=top_count.values)
        plt.title("Top 3 Manufacturers by Handset Count")
        plt.xlabel('Handset Manufacturer')
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()  # Clear the figure

    def top_5_handsets_per_top_3_handset_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3).index
        top_5_handsets = self.df[self.df['Handset Manufacturer'].isin(top_count)].groupby(
            ['Handset Manufacturer', 'Handset Type']).size().groupby('Handset Manufacturer', group_keys=False).nlargest(5)
        return top_5_handsets

    def plot_top_5_handsets_per_top_3_handset_manufacturer(self):
        self.remove_null_spe_col()
        top_count = self.df.groupby('Handset Manufacturer')['Handset Type'].count().nlargest(3).index
        filtered_df = self.df[self.df['Handset Manufacturer'].isin(top_count)]
        top_5_handsets = (filtered_df.groupby(['Handset Manufacturer', 'Handset Type'])
                        .size()
                        .groupby('Handset Manufacturer', group_keys=False)
                        .nlargest(5))
        plot_data = top_5_handsets.reset_index(name='count')
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Handset Type', y='count', hue='Handset Manufacturer', data=plot_data)
        plt.title("Top 5 Handsets by Handset Manufacturer")
        plt.xlabel('Handset Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Handset Manufacturer')
        st.pyplot(plt)
        plt.clf()  # Clear the figure

    def outlier_check_perc(self):
        st.subheader("Outlier Percentage")
        numeric_df = self.df.select_dtypes(include=[float, int])
        if numeric_df.empty:
            st.error("No numeric columns available for outlier detection.")
            return
        
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        outliers = (numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))
        total_values = numeric_df.count()
        outlier_percentage = (outliers.sum() / total_values) * 100
        outlier_df = pd.DataFrame({
            'Column': numeric_df.columns,
            'Outlier Percentage': outlier_percentage
        }).sort_values(by='Outlier Percentage', ascending=False)
        st.write(outlier_df)

    def aggregate_user_behavior(self):
        self.df.dropna(subset=['Bearer Id', 'Dur. (ms)', 'IMSI'], inplace=True)
        self.remove_outliers()
        required_columns = [
            'IMSI', 'Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
            'Social Media DL (Bytes)', 'Social Media UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)',
            'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)',
            'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Youtube DL (Bytes)', 'Youtube UL (Bytes)'
        ]
        missing_cols = [col for col in required_columns if col not in self.df.columns]
        if missing_cols:
            st.error(f"DataFrame is missing required columns: {missing_cols}")
            return
        
        aggregated_data = self.df.groupby('IMSI').agg(
            number_of_xDR_sessions=('Bearer Id', 'nunique'),
            total_session_duration=('Dur. (ms)', 'sum'),
            total_DL_data=('Total DL (Bytes)', 'sum'),
            total_UL_data=('Total UL (Bytes)', 'sum'),
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
            Youtube_Gaming_DL=('Youtube DL (Bytes)', 'sum'),
            Youtube_Gaming_UL=('Youtube UL (Bytes)', 'sum')
        ).reset_index()
        
        return aggregated_data

    def plot_insights(self):
        aggregated_data = self.aggregate_user_behavior()
        if aggregated_data is not None:
            st.subheader("Distribution of Total Download Data")
            plt.figure(figsize=(12, 8))
            sns.histplot(aggregated_data['total_DL_data'], kde=True, color='blue')
            plt.title('Distribution of Total Download Data (Bytes)')
            plt.xlabel('Total Download Data (Bytes)')
            plt.ylabel('Frequency')
            st.pyplot(plt)
            plt.clf()  # Clear the figure

    def plot_insight1(self):
        aggregated_data = self.aggregate_user_behavior()
        if aggregated_data is not None:
            st.subheader("Distribution of Total Upload Data")
            plt.figure(figsize=(12, 8))
            sns.histplot(aggregated_data['total_UL_data'], kde=True, color='red')
            plt.title('Distribution of Total Upload Data (Bytes)')
            plt.xlabel('Total Upload Data (Bytes)')
            plt.ylabel('Frequency')
            st.pyplot(plt)
            plt.clf()  # Clear the figure

    def plot_insight2(self):
        aggregated_data = self.aggregate_user_behavior()
        if aggregated_data is not None:
            st.subheader("Top 5 Handsets by Manufacturer")
            top_handsets = self.top_5_handsets_per_top_3_handset_manufacturer()
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Handset Type', y='count', hue='Handset Manufacturer', data=top_handsets.reset_index(name='count'))
            plt.title("Top 5 Handsets by Manufacturer")
            plt.xlabel('Handset Type')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            st.pyplot(plt)
            plt.clf()  # Clear the figure

    def plot_insight3(self):
        aggregated_data = self.aggregate_user_behavior()
        if aggregated_data is not None:
            st.subheader("Session Frequency Distribution")
            plt.figure(figsize=(12, 8))
            sns.histplot(aggregated_data['number_of_xDR_sessions'], kde=True, color='green')
            plt.title('Distribution of Session Frequency')
            plt.xlabel('Number of xDR Sessions')
            plt.ylabel('Frequency')
            st.pyplot(plt)
            plt.clf()  # Clear the figure

    def remove_outliers(self):
        numeric_df = self.df.select_dtypes(include=[float, int])
        if numeric_df.empty:
            st.error("No numeric columns available for outlier removal.")
            return
        
        Q1 = numeric_df.quantile(0.25)
        Q3 = numeric_df.quantile(0.75)
        IQR = Q3 - Q1
        filter = ~((numeric_df < (Q1 - 1.5 * IQR)) | (numeric_df > (Q3 + 1.5 * IQR))).any(axis=1)
        self.df = self.df[filter]

    def run_all(self):
        self.basic_info()
        self.summary_statistics()
        self.missing_percentage()
        self.data_types()
        self.plot_bar()
        self.plot_top_ten_handset_by_manufacturer()
        self.plot_top_5_handsets_per_top_3_handset_manufacturer()
        self.outlier_check_perc()
        self.plot_insights()
        self.plot_insight1()
        self.plot_insight2()
        self.plot_insight3()

