
class DataAnalyzer:
    def find_trends(self, df):
        trends = []
        for col in df.select_dtypes(include="number").columns:
            if len(df) > 1:
                if df[col].iloc[-1] > df[col].iloc[0]:
                    trends.append(f"{col} is increasing")
                else:
                    trends.append(f"{col} is decreasing")
        return trends
    
    def get_stats(self, df):
        return df.describe().to_dict()
