import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(df, fit_scaler=False, scaler=None):
    df = df.copy()
    # Encode categorical
    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    # Scale numerical
    num_cols = df.select_dtypes(include=['int64','float64']).columns
    if fit_scaler:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        return df, scaler
    else:
        df[num_cols] = scaler.transform(df[num_cols])
        return df
