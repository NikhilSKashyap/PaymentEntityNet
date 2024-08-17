import torch
from torch_geometric.data import Data
from models.paymentnet import PaymentNet
from utils.data_loader import load_data

def train(model, data, optimizer, criterion, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and prepare data
    x, edge_index, y = load_data('processed_transactions.csv')
    data = Data(x=x, edge_index=edge_index, y=y).to(device)
    
    # Initialize model
    model = PaymentNet(input_dim=29, hidden_dim=64, output_dim=32, num_heads=4, num_layers=2).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    
    # Train the model
    train(model, data, optimizer, criterion, epochs=100)
    
    # Save the model
    torch.save(model.state_dict(), 'paymentnet_model.pth')
