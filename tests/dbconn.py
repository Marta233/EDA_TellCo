import unittest
import pandas as pd
import sys
sys.path.insert(0, 'D:\10 A KAI 2\Week 2\EDA_TellCo\Scripts')
from Scripts.db_conn import Data_lolad

class TestFetchData(unittest.TestCase):
    def test_fetch_data(self):
        # Test that data is fetched and returned as a DataFrame
        df = Data_lolad()
        self.assertIsNotNone(df)
        self.assertTrue(isinstance(df, pd.DataFrame))

if __name__ == '__main__':
    unittest.main()
