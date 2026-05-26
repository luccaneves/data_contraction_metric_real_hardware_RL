import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import random
import sys
sys.path.append("..") # Adds higher directory to python modules path.

from models.models_env_trained.model_only_actuator_new_data_1.mlp_lyap_mean_bb import BB_Lyap_Mean
from mlp_lyap_mean_bb import BB_Lyap_Mean

import json
import gymnasium as gym
from gymnasium.envs.registration import register
import torch
import torch.nn as nn
import torch.nn.functional as Functional
import matplotlib.pyplot as plt

class ModelONNX(nn.Module):
    def __init__(self, hidden_size = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # 8 means + 8 stds
        )

    def forward(self, x):
        return self.net(x)  # or torch.cat([action, self.log_std_parameter.expand_as(action)], dim=-1)
    

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from '{path}' at epoch {checkpoint['epoch']}")
    return checkpoint


def main():
    # =======================
    # 🔧 Configuration
    # =======================

    checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/models_env_trained/model_only_actuator_new_data_1/model_80.pth'
    checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/train_model/runs/2025-10-09_20-01-09/model_390.pth'

    dataset_csv = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/experimentsdata_normalized.csv'
    dataset_csv= "C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/20k_random_eval_new.csv"
    
    json_path = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/experimentsdata_normalizing_constants.json'

    with open(json_path, 'r') as file:
        normalizing_constants = json.load(file)

    get_norm = lambda key: (
        normalizing_constants[key]['min'],
        normalizing_constants[key]['max']
    )

    keys = ['F_actuator', 'F_actuator_deriv', 'F_load', 'F_load_deriv',
                'Pa', 'Pb', 'x', 'x_deriv']  # last_x shares x scale
        
    mins = [get_norm(k)[0] for k in keys]
    maxs = [get_norm(k)[1] for k in keys]


    n_steps = 6*5000


    # =======================
    # ⚙️ Setup
    # =======================
    device = get_device()

    data = pd.read_csv(dataset_csv)
    max_start = len(data) - n_steps
    if max_start <= 0:
        raise ValueError("Dataset is too small for the specified number of steps.")

    start_idx = random.randint(0, max_start)
    print(start_idx)
    start_idx = 5000
    end_idx = start_idx + n_steps

    print(data.iloc[start_idx, 0])

    print(f"🔹 Starting at index {start_idx} → Validating {n_steps} steps.")

    input_i = data.iloc[start_idx:end_idx, 9].values  # Input signal 'i'
    x = data.iloc[start_idx:end_idx, 7].values  # Input signal 'i'
    dx = data.iloc[start_idx:end_idx, 8].values  # Input signal 'i'
    Fl = data.iloc[start_idx:end_idx, 3].values  # Input signal 'i'
    dFl = data.iloc[start_idx:end_idx, 4].values  # Input signal 'i'
    dFh = data.iloc[start_idx:end_idx, 2].values  # Input signal 'i'
    Pa = data.iloc[start_idx:end_idx, 5].values  # Input signal 'i'
    Pb = data.iloc[start_idx:end_idx, 6].values  # Input signal
    true_states = data.iloc[start_idx:end_idx, [1]].values  # True states

    # =======================
    # 🚀 Load Model
    # =======================

    model = BB_Lyap_Mean(hidden_size = 96,ts = 0.001, json_file_path=json_path).to(device)
    load_checkpoint(checkpoint_path, model)
    model.eval()



    # =======================
    # 🔗 Initialize First State
    # =======================
    current_state = data.iloc[start_idx, [1,2,5,6,7,8,9]].values
    current_i = data.iloc[start_idx, 9]

    current_input = np.concatenate([current_state])
    current_input = torch.tensor(current_input, dtype=torch.float32).unsqueeze(0).to(device)

    state_analytic = current_input
    state_bb = current_input

    predicted_trajectory = []
    predicted_trajectory_analytic = []

    predicted_trajectory_not_normalized = []
    predicted_trajectory_analytic_not_normalized = []

    V_values = []
    V_deltas = []

    f = []
    g = []
    i = []
    f_g = []

    # =======================
    # 🔄 Propagate
    # =======================
    with torch.no_grad():
        for step in range(n_steps):
            #print(current_input)
   
            output_bb = model.single_step(2*state_bb - 1)  # (1, 8)
            output_bb = output_bb/2 + 0.5
            output_bb = state_bb[0, 0:1] + output_bb*0.001



            f.append(model.net_1(2*state_bb[:,0:6] - 1).item()*(2*state_bb[:,6]-1).item())
            g.append(0)

            state = torch.cat([
                output_bb[0, 0:1]
            ], dim=0)       

            predicted_trajectory.append(state.cpu().numpy().flatten())    

            if step + 1 < n_steps:
                next_i = input_i[step + 1]
                next_x = x[step + 1]
                next_x_deriv = dx[step + 1]
                next_F_load = Fl[step + 1]
                next_dF_load = dFl[step + 1]
                next_dF_hyd = dFh[step + 1]
                next_Pa = Pa[step + 1]
                next_Pb = Pb[step + 1]


                state_bb = torch.cat([
                    torch.tensor([output_bb[0].item()], dtype=torch.float32, device=device),
                    torch.tensor([next_dF_hyd], dtype=torch.float32, device=device),
                    torch.tensor([next_Pa], dtype=torch.float32, device=device),
                    torch.tensor([next_Pb], dtype=torch.float32, device=device),
                    torch.tensor([next_x], dtype=torch.float32, device=device),
                    torch.tensor([next_x_deriv], dtype=torch.float32, device=device),
                    torch.tensor([next_i], dtype=torch.float32, device=device)
                ]).unsqueeze(0)  # (1, 9)

    


    predicted_trajectory = np.array(predicted_trajectory)  # (n_steps, 8)
   

    predicted_trajectory_analytic = np.array(predicted_trajectory_analytic)  # (n_steps, 8)

    # =======================
    # 📊 Metrics
    # =======================
    mse1 = mean_squared_error(true_states[:,0], predicted_trajectory[:,0])
 



    print(f"\n✅ Validation Results (Steps {start_idx} to {end_idx}):")
    print(f"🔹 F_actuator: {mse1:.6f}")




    # =======================
    # 📈 Plot
    # =======================
    state_labels = ['$f_h$', '$\dot{f_h}$', '$p_a$', '$p_b$']
    
    plt.figure(figsize=(16, 10))
    for i in range(1):
        plt.subplot(1, 1, i + 1)
        plt.plot(true_states[:, i], label="True", linewidth=2)
        plt.plot(predicted_trajectory[:, i], label="Predicted", linestyle='--')
        plt.title(state_labels[i],fontsize= 18)
        plt.xlabel('Time (ms)',fontsize='large')
        plt.ylabel('')
        plt.legend()
        plt.ylim(0, 1.3)
        plt.grid()


    plt.tight_layout()
    plt.show()

    plt.figure(2)
    plt.plot(f)

    plt.figure(3)
    plt.plot(g)


    plt.show()

    net_1_onnx = ModelONNX(hidden_size=96)
    net_2_onnx = ModelONNX(hidden_size=96)
    net_1_onnx.net.load_state_dict(model.net_1.state_dict(), strict=True)
    net_2_onnx.net.load_state_dict(model.net_1.state_dict(), strict=True)

    dummy_input = torch.randn(1, 9)

    torch.onnx.export(
        net_1_onnx,
        dummy_input,
        "model_1.onnx",
        export_params=True,
        opset_version=13,                 # Use opset 13 (safe range for MATLAB)
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Remove dynamic axes (not MATLAB-friendly)
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    dummy_input = torch.randn(1, 9)

    torch.onnx.export(
        net_2_onnx,
        dummy_input,
        "model_2.onnx",
        export_params=True,
        opset_version=13,                 # Use opset 13 (safe range for MATLAB)
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        # Remove dynamic axes (not MATLAB-friendly)
        # dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    )

    from scipy.io import savemat

    metric_pt = net_1_onnx.net.to("cpu").eval()  # <— ORIGINAL net

    # Known input in the metric's coordinate system (8 features)
    # Use values in the SAME normalized space the metric expects.
    X_CB = np.array([[ 0.10],
                    [-0.20],
                    [ 0.30],
                    [-0.40],
                    [ 0.50],
                    [ 0.00],
                    [ 0.70],
                    [-0.80],
                    [ 0.10]], dtype=np.float32)   # shape (8,1), 'CB' = Channels x Batch

    # PyTorch forward is batch-first => (1,8)
    x_pt = torch.from_numpy(X_CB.T.copy())      # (1, 8)

    with torch.no_grad():
        Y_pt = metric_pt(x_pt).cpu().numpy()    # (1, 64)

    # Save for MATLAB in 'CB' shapes: input (8x1), output (64x1)
    savemat("model_1_known_pair.mat", {
        "X_CB": X_CB.astype(np.single),                 # (8,1)
        "Y_metric_pt_CB": Y_pt.T.astype(np.single)      # (64,1)
    })

    metric_pt = net_2_onnx.net.to("cpu").eval()  # <— ORIGINAL net

    # Known input in the metric's coordinate system (8 features)
    # Use values in the SAME normalized space the metric expects.
    X_CB = np.array([[ 0.10],
                    [-0.20],
                    [ 0.30],
                    [-0.40],
                    [ 0.50],
                    [ 0.00],
                    [ 0.70],
                    [-0.80],
                    [ 0.10]], dtype=np.float32)   # shape (8,1), 'CB' = Channels x Batch

    # PyTorch forward is batch-first => (1,8)
    x_pt = torch.from_numpy(X_CB.T.copy())      # (1, 8)

    with torch.no_grad():
        Y_pt = metric_pt(x_pt).cpu().numpy()    # (1, 64)

    # Save for MATLAB in 'CB' shapes: input (8x1), output (64x1)
    savemat("model_2_known_pair.mat", {
        "X_CB": X_CB.astype(np.single),                 # (8,1)
        "Y_metric_pt_CB": Y_pt.T.astype(np.single)      # (64,1)
    })
    
    




if __name__ == "__main__":
    main()