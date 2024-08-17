import pandas as pd
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

def load_data(file_path, test_size=0.2):
    df = pd.read_csv(file_path)
    
    # Prepare features
    features = df[['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']].values
    features = torch.FloatTensor(features)
    
    # Prepare labels (fraud/non-fraud)
    labels = torch.LongTensor(df['Class'].values)
    
    # Create edge index (connections between transactions)
    edge_index = []
    for merchant in df['merchant_id'].unique():
        merchant_transactions = df[df['merchant_id'] == merchant].index.tolist()
        edge_index.extend([(i, j) for i in merchant_transactions for j in merchant_transactions if i != j])
    edge_index = torch.LongTensor(edge_index).t().contiguous()
    
    # Create PyTorch Geometric Data object
    data = Data(x=features, edge_index=edge_index, y=labels)
    
    # Split data into train and test sets
    num_samples = len(df)
    train_indices, test_indices = train_test_split(range(num_samples), test_size=test_size, stratify=labels)
    
    train_data = Data(x=features[train_indices], 
                      edge_index=edge_index[:, edge_index[0].isin(train_indices) & edge_index[1].isin(train_indices)],
                      y=labels[train_indices])
    
    test_data = Data(x=features[test_indices], 
                     edge_index=edge_index[:, edge_index[0].isin(test_indices) & edge_index[1].isin(test_indices)],
                     y=labels[test_indices])
    
    return train_data, test_data
