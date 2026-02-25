import torch
import numpy as np
from src.competitors.mmd_fair.simulation.fair_fl_datasets import CompasDataset
from src.competitors.mmd_fair.simulation.fair_fl_models import TwoLayerNN
from src.competitors.mmd_fair.simulation.fair_fl_experiment import FairFLServer, accuracy, P1

def main():
    torch.manual_seed(0)
    np.random.seed(0)
    
    datasets = CompasDataset().load_data(homefolder="/home/lcorbucci/Fair-FL")
    input_size = 8
    
    def model_factory():
        return TwoLayerNN(input_size)
        
    s = FairFLServer(
        datasets,
        model_factory,
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
    
    global_acc = accuracy(all_preds, all_labels)
    global_p1 = P1(all_preds, all_sensitive)
    
    print(f"Global Accuracy: {global_acc:.4f}")
    print(f"Global SP Unfairness: {global_p1:.4f}")
    
    local_p1_mean = np.mean([P1(p, a) for p,y,a in res])
    print(f"Local SP Unfairness Mean: {local_p1_mean:.4f}")

if __name__ == "__main__":
    main()
