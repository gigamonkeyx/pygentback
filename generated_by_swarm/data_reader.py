
import pandas as pd

class DataReader:
    def read_csv(self, filename):
        return pd.read_csv(filename)
    
    def get_info(self, df):
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "types": str(df.dtypes.to_dict())
        }
