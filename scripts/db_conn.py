import psycopg2
import pandas as pd
from dotenv import load_dotenv
import os

class DataLoader:
    def __init__(self):
        self.conn = psycopg2.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST'),
            port=os.getenv('DB_PORT')
        )

    def fetch_data(self):
        # SQL query to fetch data from xdr_data table
        query = "SELECT * FROM xdr_data;"
        # Load data into pandas DataFrame
        df = pd.read_sql(query, self.conn)
        # Close connection after loading data
        self.conn.close()
        return df
