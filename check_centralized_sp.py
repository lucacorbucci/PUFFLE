import sys
import torch
import numpy as np

sys.path.append("/home/lcorbucci/Fair-FL")
from src.utils.datasets import CompasDataset
from src.centralized import train as centralized_train
from src.utils import metrics

def buildModelClass(inputSize):
    class NNModel(torch.nn.Module):
        def __init__(self, inputSize=inputSize, outputSize=1):
            super(NNModel, self).__init__()
            self.linear1 = torch.nn.Linear(inputSize, 16, bias=True)
            self.linear2 = torch.nn.Linear(16, outputSize, bias=True)

        def forward(self, x):
            x = torch.nn.functional.relu(self.linear1(x))
            out = self.linear2(x)
            return out
    return NNModel

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    
    datasets = CompasDataset().load_data(HOMEFOLDER="/home/lcorbucci/Fair-FL")
    
    # Centralized model trains on concatenated data
    X_train, Y_train, A_train = [], [], []
    for (x, y, a) in datasets:
        X_train.append(torch.tensor(x.to_numpy(), dtype=torch.float32))
        Y_train.append(torch.tensor(y.to_numpy(), dtype=torch.float32))
        A_train.append(torch.tensor(a.to_numpy(), dtype=torch.float32))
        
    X = torch.cat(X_train)
    Y = torch.cat(Y_train)
    A = torch.cat(A_train)
    
    # split
    from sklearn.model_selection import train_test_split
    X_tr, X_te, Y_tr, Y_te, A_tr, A_te = train_test_split(X, Y, A, test_size=0.25)
    
    model = buildModelClass(8)()
    optimizer = torch.optim.SGD(model.parameters(), lr=5e-2)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # train without fairness constraints (agnostic)
    dataset_tr = torch.utils.data.TensorDataset(X_tr, A_tr, Y_tr)
    dataloader = torch.utils.data.DataLoader(dataset_tr, batch_size=100, shuffle=True)
    
    for epoch in range(100):
        for bx, ba, by in dataloader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out.view(-1), by)
            loss.backward()
            optimizer.step()
            
    with torch.no_grad():
        out_te = model(X_te).view(-1)
        acc = metrics.accuracy(out_te, Y_te)
        p1 = metrics.P1(out_te, A_te)
        
    print(f"Centralized Accuracy: {acc:.4f}")
    print(f"Centralized SP Unfairness: {p1:.4f}")

if __name__ == "__main__":
    main()
