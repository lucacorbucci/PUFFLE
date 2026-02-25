import sys
import torch
import numpy as np

sys.path.append("/home/lcorbucci/Fair-FL")
from src.utils.datasets import CompasDataset
from src.ours import server as our_server
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
    
    s = our_server.Server(
        datasets,
        buildModelClass(8),
        torch.nn.BCEWithLogitsLoss(),
        m=None,
        T=100,  # Just do 100 rounds
        mu=1.0,
        NY=100,
        lambda_=1e-5,  # Smallest lambda -> most unfairness
        datasetname="Compas",
        runname="test_global_p1",
        device="cpu",
    )
    
    s.train_test_split()
    s.sync_N()
    s.sync_Pa()
    s.train()
    
    res, weights = s.test_current_model()
    
    all_preds = torch.cat([r[0] for r in res]).flatten()
    all_labels = torch.cat([r[1] for r in res]).flatten()
    all_sensitive = torch.cat([r[2] for r in res]).flatten()
    
    global_acc = metrics.accuracy(all_preds, all_labels).cpu()
    global_p1 = metrics.P1(all_preds, all_sensitive).cpu()
    
    print(f"Fair-FL Global Accuracy: {global_acc:.4f}")
    print(f"Fair-FL Global SP Unfairness: {global_p1:.4f}")
    
    local_p1_mean = np.mean([metrics.P1(p, a).cpu() for p,y,a in res])
    print(f"Fair-FL Local SP Unfairness Mean: {local_p1_mean:.4f}")

if __name__ == "__main__":
    main()
