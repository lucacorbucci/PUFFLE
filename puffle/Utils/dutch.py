from torch.utils.data import Dataset


class TabularDataset(Dataset):
    def __init__(self, x, z, w, y):
        """
        Initialize the custom dataset with x (features), z (sensitive values), and y (targets).

        Args:
        x (list of tensors): List of input feature tensors.
        z (list): List of sensitive values.
        y (list): List of target values.
        """
        self.samples = x
        self.sensitive_features = z
        self.sensitive_features_2 = w
        self.targets = y
        self.indexes = range(len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get a single data point from the dataset.

        Args:
        idx (int): Index to retrieve the data point.

        Returns:
        sample (dict): A dictionary containing 'x', 'z', and 'y'.
        """
        x_sample = self.samples[idx]
        z_sample = self.sensitive_features[idx]
        w_sample = self.sensitive_features_2[idx]
        y_sample = self.targets[idx]

        return x_sample, z_sample, w_sample, y_sample
