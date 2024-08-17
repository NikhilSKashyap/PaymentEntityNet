from transformers import AutoModel, AutoTokenizer
import torch.nn as nn

class TransactionTransformer(nn.Module):
    def __init__(self, model_name='distilbert-base-uncased'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.fc = nn.Linear(self.model.config.hidden_size, 64)

    def forward(self, x):
        # Assume x is a batch of transaction descriptions
        inputs = self.tokenizer(x, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model(**inputs)
        return self.fc(outputs.last_hidden_state[:, 0, :])  # Use [CLS] token
