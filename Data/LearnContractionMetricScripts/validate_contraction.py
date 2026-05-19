import torch

import sys
sys.path.append("..") # Adds higher directory to python modules path.

from torch.utils.data import DataLoader

from dataset_lyap import HydraulicDataset
from models.models_contraction_metric_trained.contraction_metric_new_data_1.MetricNet import MetricNet

from models.models_env_trained.model_new_data_20K_1.mlp_lyap_mean_bb import BB_Lyap_Mean
from models.models_contraction_metric_trained.contraction_metric_new_data_1.train_metric_new import contraction_certificate_value_Horizon, controller
import json

import torch
import cvxpy as cp
import numpy as np


# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

json_path_model = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/' \
'experimentsdata_20k_normalizing_constants.json'

checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/models_env_trained/model_new_data_20K_1' \
'/model_380.pth'

csv_path = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/' \
'stable_traj_new_data_horizon.csv'

metric_checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/' \
'models/models_contraction_metric_trained/contraction_metric_new_data_1/model_6650.pth'

#300

# Load dynamics model
model = BB_Lyap_Mean(hidden_size=64,ts = 0.001, json_file_path=json_path_model).to(device)
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load trained metric
state_dim = 8
M_net = MetricNet(state_dim, hidden_dim=32).to(device)
metric_ckpt = torch.load(metric_checkpoint_path, map_location=device)
M_net.load_state_dict(metric_ckpt['model_state_dict'])
M_net.eval()



# Load dataset
dataset = HydraulicDataset(csv_file=csv_path)

subset_ratio = 0.01  # Use 10% of the dataset
subset_size = int(len(dataset) * subset_ratio)

from torch.utils.data import Subset
import numpy as np

subset_indices = np.random.choice(len(dataset), size=subset_size, replace=False)
subset = Subset(dataset, subset_indices)

dataloader = DataLoader(subset, batch_size=1, shuffle=False)



# Store results
results = {
    'cert_stable': [],
    'cert_unstable': []
}

# Evaluate contraction metric
def print_debug(debug):
    print("M:")
    print(debug["M"])

    print("M_dot:")
    print(debug["M_dot"])

    print("A:")
    print(debug["A"])

    print("B:")
    print(debug["B"])

    print("K:")
    print(debug["K"])

    print("BK:")
    print(debug["BK"])

    print("sym_term:")
    print(debug["sym_term"])

    print("contraction_matrix:")
    print(debug["contraction_matrix"])

    print("v:")
    print(debug["v"])

with torch.no_grad():
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        x = x_batch[:, :8]
        u = x_batch[:, 10]
        ref = x_batch[:, 11]
        Kp = x_batch[:, 12]
        Ki = x_batch[:, 13]
        integral = x_batch[:, 8]
        last_ref = x_batch[:, 9]
        is_stable = x_batch[:, -1]

        cert, debug = contraction_certificate_value_Horizon(
            x, u, ref, integral, last_ref, model.single_step, M_net, controller,is_stable,Kp,Ki,
            delta=1e-4, lambda_val=1, task_dim=2
        )

        contraction_matrix = debug["contraction_matrix"]


        #print_debug(debug)
        #print("cert:")
        #print(cert)

        if(np.all(np.linalg.eigvals(debug["M"].cpu().detach().numpy()) > 0) == False):
            print("Eig Error")

        is_stable = x_batch[:, -1] > 0.5
        epsilon = 1e-3
        cert[cert.abs() < epsilon] = 0.0
        
        results['cert_stable'].append(cert[is_stable].cpu())
        results['cert_unstable'].append(cert[~is_stable].cpu())

# Concat and print statistics
cert_stable = torch.cat(results['cert_stable'])
cert_unstable = torch.cat(results['cert_unstable'])

print("✅ Contraction certificate statistics:")
if cert_stable.numel() > 0:
    print(f"  Stable   - mean: {cert_stable.mean():.10e}, max: {cert_stable.max():.10e}, min: {cert_stable.min():.10e}, >0 ratio: {(cert_stable > 0).float().mean():.10%}")
else:
    print("  stable - No stable samples found in dataset.")


if cert_unstable.numel() > 0:
    print(f"  Unstable - mean: {cert_unstable.mean():.10e}, max: {cert_unstable.max():.10e}, min: {cert_unstable.min():.10e}, <0 ratio: {(cert_unstable < 0).float().mean():.10%}")
else:
    print("  Unstable - No unstable samples found in dataset.")