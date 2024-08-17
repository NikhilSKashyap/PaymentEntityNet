import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def download_data():
    # Using the Credit Card Fraud Detection dataset from Kaggle
    # You would need to download this manually due to Kaggle's terms
    df = pd.read_csv('creditcard.csv')
    return df

def preprocess_data(df):
    # Normalize numerical features
    scaler = StandardScaler()
    df.iloc[:, 1:29] = scaler.fit_transform(df.iloc[:, 1:29])
    
    # Create synthetic merchant and issuer IDs (not in original dataset)
    df['merchant_id'] = np.random.randint(1, 1000, df.shape[0])
    df['issuer_id'] = np.random.randint(1, 100, df.shape[0])
    def create_description(row):
        amount = row['Amount']
        merchant = f"Merchant_{row['merchant_id']}"
        return f"Transaction of ${amount:.2f} at {merchant}"

    df['description'] = df.apply(create_description, axis=1)

    return df

if __name__ == "__main__":
    raw_data = download_data()
    processed_data = preprocess_data(raw_data)
    processed_data.to_csv('processed_transactions.csv', index=False)