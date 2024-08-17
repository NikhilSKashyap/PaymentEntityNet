from transformers import AutoConfig
import torch
import torch.nn as nn
from .transformer import TransactionTransformer
from .gnn import GraphSAGE

class PaymentNet(nn.Module):
    def __init__(self, num_numerical_features, hidden_dim, output_dim, model_name='distilbert-base-uncased'):
        super().__init__()
        self.transformer = TransactionTransformer(model_name)
        config = AutoConfig.from_pretrained(model_name)
        self.gnn = GraphSAGE(config.hidden_size + num_numerical_features, hidden_dim, output_dim)
        self.fc = nn.Linear(output_dim * 2, output_dim)

    def forward(self, numerical_features, descriptions, edge_index):
        trans_emb = self.transformer(descriptions)
        combined_features = torch.cat([numerical_features, trans_emb], dim=1)
        graph_emb = self.gnn(combined_features, edge_index)
        combined = torch.cat([trans_emb, graph_emb], dim=1)
        return self.fc(combined)
