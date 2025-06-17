def load_data(path):
    """Load dataset from file"""
    import pandas as pd
    return pd.read_csv(path)
