import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


class SmartPixelLocalDataset(Dataset):
    """
    Custom Dataset for Smart-Pixel local CSV data.
    Expects separate feature and label CSV files.
    """

    def __init__(self, x_csv, y_csv):
        if not os.path.exists(x_csv) or not os.path.exists(y_csv):
            raise FileNotFoundError(f"Missing dataset file(s): {x_csv}, {y_csv}")

        df_X = pd.read_csv(x_csv)
        df_y = pd.read_csv(y_csv)

        self.X = df_X.values.astype("int8")
        self.y = df_y.values.squeeze().astype("int8") # !: check type

        assert len(self.X) == len(self.y), "Feature-label size mismatch!"

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx])
        y = torch.tensor(self.y[idx])
        return x, y

def get_smartpixel_dataloaders(
    data_root: str,
    num_workers: int = 0,
    batch_size_train: int = 64,
    batch_size_test: int = 1024,
    pin_memory: bool = True,
    pin_memory_device: str = "cpu",
    local_id: int = 0,
):
    """
    Load Smart-Pixel local dataset and return DataLoaders.
    Each local dataset includes 4 CSVs:
      - QuantizedInputTrainSetLocal{local_idx}.csv
      - TrainSetLabelLocal{local_idx}.csv
      - QuantizedInputTestSetLocal{local_idx}.csv
      - TestSetLabelLocal{local_idx}.csv
    """
    train_X_path = os.path.join(data_root, f"QuantizedInputTrainSetLocal{local_id}.csv")
    train_y_path = os.path.join(data_root, f"TrainSetLabelLocal{local_id}.csv")
    test_X_path = os.path.join(data_root, f"QuantizedInputTestSetLocal{local_id}.csv")
    test_y_path = os.path.join(data_root, f"TestSetLabelLocal{local_id}.csv")

    train_dataset = SmartPixelLocalDataset(train_X_path, train_y_path)
    test_dataset = SmartPixelLocalDataset(test_X_path, test_y_path)

    print(f"Local {local_id}:")
    print("  Training set shape :", len(train_dataset), "samples")
    print("  Test set shape     :", len(test_dataset), "samples")


    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_test,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    return train_loader, test_loader

# New function to get global dataloaders by merging multiple local datasets
def get_global_smartpixel_dataloaders(
    data_root: str,
    local_ids=range(12),  # merge Local0–11
    num_workers: int = 12,
    batch_size_train: int = 64,
    batch_size_test: int = 1024,
    pin_memory: bool = True,
    pin_memory_device: str = "cpu",
):
    datasets_train = []
    datasets_test = []

    for local_id in local_ids:
        train_X_path = os.path.join(data_root, f"QuantizedInputTrainSetLocal{local_id}.csv")
        train_y_path = os.path.join(data_root, f"TrainSetLabelLocal{local_id}.csv")
        test_X_path = os.path.join(data_root, f"QuantizedInputTestSetLocal{local_id}.csv")
        test_y_path = os.path.join(data_root, f"TestSetLabelLocal{local_id}.csv")

        datasets_train.append(SmartPixelLocalDataset(train_X_path, train_y_path))
        datasets_test.append(SmartPixelLocalDataset(test_X_path, test_y_path))

    # Combine all locals into one dataset
    merged_train = ConcatDataset(datasets_train)
    merged_test = ConcatDataset(datasets_test)

    print(f"Merged {len(local_ids)} local datasets into global loaders.")
    print("  Training samples:", len(merged_train))
    print("  Test samples:", len(merged_test))

    train_loader = DataLoader(
        merged_train,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    test_loader = DataLoader(
        merged_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        pin_memory_device=pin_memory_device,
    )

    return train_loader, test_loader