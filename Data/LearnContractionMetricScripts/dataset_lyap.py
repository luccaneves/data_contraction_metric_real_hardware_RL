import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

# Custom dataset
class HydraulicDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Tag each row with a "run id"
        self.data['run_id'] = (self.data['t'] == 0.0).cumsum()

        # Precompute valid indices (not the last of a run)
        self.valid_indices = self.data.index[
            self.data['run_id'].shift(-1) == self.data['run_id']
        ].tolist()
        
        # Base variables
        base_vars = ["F_actuator", "F_actuator_deriv", "F_load", "F_load_deriv", "Pa", "Pb", "x", 
                     "x_deriv","integral_error","last_desired_force","i","Ref","Kp","Ki","lambda"]
        predict_horizon = 10

        # Build features
        self.features = base_vars

        for i in range(1, predict_horizon + 1):
            self.features += [
                f"F_actuator_{i}",
                f"F_actuator_deriv_{i}",
                f"F_load_{i}",
                f"F_load_deriv_{i}",
                f"Pa_{i}",
                f"Pb_{i}",
                f"x_{i}",
                f"x_deriv_{i}",
                f"integral_error_{i}",
                f"last_desired_force_{i}",
                f"i_{i}",
                f"Ref_{i}",
                f"Kp_{i}",
                f"Ki_{i}",
                f"lambda_{i}",
            ]

        self.features += ["Stable"]        

        #print(self.features)

        #Build targets (without 'i' and 'last_x' unless needed)
        self.target = []

        for i in range(1, predict_horizon + 1):
            self.target += [
                f"F_actuator_{i}"
            ]


        self.X = self.data[self.features].values.astype(np.float32)
        self.Y = self.data[self.target].values.astype(np.float32)
        self.index_map = {idx: i for i, idx in enumerate(self.data.index)}

    def __len__(self):
        return len(self.valid_indices)

    def get_num_features(self):
        # Subtract one for target column
        return len(self.features)
    
    def get_num_target(self):
        # Subtract one for target column
        return len(self.target)

    def get_features_names(self):
        return self.features
    
    def get_target_names(self):
        return self.target

    def __getitem__(self, idx):
        row_idx = self.index_map[self.valid_indices[idx]]
        x = torch.from_numpy(self.X[row_idx])
        y = torch.from_numpy(self.Y[row_idx])
        return x, y

if __name__ == "__main__":
    # Example usage
    dataset = HydraulicDataset("/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalized.csv")
    print(len(dataset))
    print(dataset[0])
    print(dataset.get_num_features())