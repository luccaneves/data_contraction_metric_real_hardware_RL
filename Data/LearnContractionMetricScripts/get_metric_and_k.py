import cvxpy as cp
import numpy as np

import numpy as np

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from scipy.io import savemat
import os



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

from MetricNet import MetricNet
import numpy as np
import json

import torch
from torch.utils.data import DataLoader
from dataset_lyap import HydraulicDataset
import numpy as np
import json
from models.models_env_trained.model_new_data_20K_1.mlp_lyap_mean_bb import BB_Lyap_Mean
import numpy as np
import cvxpy as cp

# ----- finite diffs (as before) -----
def fd_jacobian_wrt_x(f_model_np, x, u, h=1e-4):
    n = x.size
    A = np.zeros((n,n))
    for j in range(n):
        xf, xb = x.copy(), x.copy()
        xf[j] += h; xb[j] -= h
        ff = f_model_np(xf, u); fb = f_model_np(xb, u)
        A[:, j] = (ff - fb)/(2*h)
    return A

def fd_jacobian_wrt_u(f_model_np, x, u, h=1e-4):
    n, m = x.size, u.size
    B = np.zeros((n,m))
    for j in range(m):
        uf, ub = u.copy(), u.copy()
        uf[j] += h; ub[j] -= h
        ff = f_model_np(x, uf); fb = f_model_np(x, ub)
        B[:, j] = (ff - fb)/(2*h)
    return B

def mema_ccm_fd_affine(f_model_np, B_fun_np, x_samples, u_ref,
                       lambda_val=0.3, alpha=1e-3, beta=1e3,
                       h=1e-4, solver="SCS", verbose=True):
    """
    Affine W(x) = W0 + sum_j Wj x_j,  Y(x) = Y0 + sum_j Yj x_j
    Finite-difference A,B. Solves CCM LMI on sampled states.
    """
    x_samples = np.asarray(x_samples, float)
    N, n = x_samples.shape
    u_ref = np.asarray(u_ref, float)
    m = u_ref.size

    # Decision matrices
    W0 = cp.Variable((n,n), symmetric=True)
    Y0 = cp.Variable((m,n))
    Wj = [cp.Variable((n,n), symmetric=True) for _ in range(n)]
    Yj = [cp.Variable((m,n)) for _ in range(n)]

    cons = []

    # Conditioning at samples: alpha I <= W(x_i) <= beta I
    I = np.eye(n)

    for i in range(N):
        x = x_samples[i]
        # Evaluate affine W(x), Y(x)
        Wxi = W0
        Yxi = Y0
        for j in range(n):
            Wxi = Wxi + Wj[j]*x[j]
            Yxi = Yxi + Yj[j]*x[j]

        # Lie derivative of W: dotW = sum_j Wj * f_j(x,u_ref)
        f_val = f_model_np(x, u_ref)
        dotW = 0
        for j in range(n):
            dotW = dotW + Wj[j]*f_val[j]

        # A,B via finite differences (unless B_fun_np provided)
        A = fd_jacobian_wrt_x(f_model_np, x, u_ref, h=h)
        B = B_fun_np(x)
        if B is None:
            B = fd_jacobian_wrt_u(f_model_np, x, u_ref, h=h)

        # CCM LMI at sample
        M = -dotW + A@Wxi + Wxi@A.T + B@Yxi + Yxi.T@B.T + 2*lambda_val*Wxi
        cons += [M << 0,  Wxi >> alpha*I,  Wxi << beta*I]

    # Optional: mild regularizer (keeps scaling reasonable)
    obj = cp.Minimize(cp.trace(W0))
    prob = cp.Problem(obj, cons)
    prob.solve(solver=solver, verbose=verbose)

    if prob.status not in ["optimal","optimal_inaccurate"]:
        if verbose: print("Infeasible CCM (try smaller λ or enrich basis/samples).")
        return None

    # Pack results
    W_affine = {"W0": W0.value, "Wj": [Wj[k].value for k in range(n)]}
    Y_affine = {"Y0": Y0.value, "Yj": [Yj[k].value for k in range(n)]}
    return W_affine, Y_affine



checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/models_env_trained/model_new_data_20K_1/' \
'model_380.pth'

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"✅ Loaded checkpoint from '{path}' at epoch {checkpoint['epoch']}")
    return checkpoint

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def B_fun_np(x):
    # If B is constant known matrix:
    return None  # np.array of shape (n,m)
    # Or return None to estimate via fd_jacobian_wrt_u
    # return None

device = "cuda" if torch.cuda.is_available() else "cpu"
model = BB_Lyap_Mean(hidden_size=64, ts = 0.001, json_file_path="json_path_model").to(device)
load_checkpoint(checkpoint_path, model)
model.eval()

# --- replace your model_fcn with this ---
def f_model_np(x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """
    Two-argument wrapper: (x, u) -> xdot
    Converts numpy -> torch, applies your [-1,1] scaling, calls model.single_step,
    then converts back to numpy (1D shape (n,)).
    """
    x = np.asarray(x, dtype=np.float32)
    u = np.asarray(u, dtype=np.float32)
    xu = np.concatenate([x, u], axis=0)            # (n+m,)

    xt = torch.from_numpy(xu).to(device).unsqueeze(0)  # (1, n+m) on CPU/GPU
    xt = 2.0 * xt - 1.0                              # your scaling to [-1, 1]

    with torch.no_grad():
        xdot_t = model.single_step(xt)               # expect (1, n)
        xdot_t = 0.5 * xdot_t + 0.5                  # back to [0, 1] as in your code

    xdot = xdot_t.squeeze(0).detach().cpu().numpy().astype(np.float32)  # (n,)
    return xdot


# Build grid
x_lo, x_hi, n = 0, 1.0, 8
N = 300  # number of samples
x_samples = np.random.uniform(x_lo, x_hi, size=(N, n))

W, Y = mema_ccm_fd_affine(f_model_np, B_fun_np, x_samples, u_ref=np.zeros((1,)), lambda_val=0.5, solver="SCS")

def evaluate_WY_affine(x: np.ndarray, W_affine: dict, Y_affine: dict):
    """
    Evaluate W(x) and Y(x) for the affine parameterization:
        W(x) = W0 + sum_j Wj * x[j]
        Y(x) = Y0 + sum_j Yj * x[j]
    """
    x = np.asarray(x, dtype=float)
    W = W_affine["W0"].copy()
    Y = Y_affine["Y0"].copy()
    for j, xj in enumerate(x):
        W += W_affine["Wj"][j] * xj
        Y += Y_affine["Yj"][j] * xj
    # Symmetrize W for numerical safety
    W = 0.5 * (W + W.T)
    return W, Y


def compute_K_task(x: np.ndarray,
                   W_affine: dict,
                   Y_affine: dict,
                   task_idx: int = 2,
                   spd_floor: float = 1e-8):
    """
    Compute:
      - full K(x) = Y(x) @ inv(W(x))
      - task-projected K_task(x) = K(x) @ P,  where P selects the desired state.
    'task_idx' is 0-based (e.g., 2 means the 3rd state).
    """
    W, Y = evaluate_WY_affine(x, W_affine, Y_affine)

    # Ensure invertibility with a tiny SPD floor if needed
    # (safer than np.linalg.inv on nearly-singular W)
    n = W.shape[0]
    W_reg = W + spd_floor * np.eye(n)
    K_full = Y @ np.linalg.solve(W_reg, np.eye(n))

    # Projection P keeps only the desired coordinate
    P = np.zeros((n, n))
    P[task_idx, task_idx] = 1.0

    K_task = K_full @ P
    return K_full, K_task


print("W =", W)
print("Y =", Y)

# After: W, Y = mema_ccm_fd_affine(...)

# -------------------------------------------------------------------
# 💾 Save the CCM results (W_affine, Y_affine) for MATLAB
# -------------------------------------------------------------------
if W is not None and Y is not None:
    save_dir = "./saved_ccm"  # customize if needed
    os.makedirs(save_dir, exist_ok=True)

    # Convert nested dicts (with lists of arrays) into MATLAB-compatible dicts
    W_mat = {
        "W0": W["W0"],
        "Wj": np.stack(W["Wj"], axis=0)  # shape (n, n, n)
    }
    Y_mat = {
        "Y0": Y["Y0"],
        "Yj": np.stack(Y["Yj"], axis=0)  # shape (n, m, n) or (n, n, m)
    }

    # Save as .mat for MATLAB
    W_path = os.path.join(save_dir, "W_affine.mat")
    Y_path = os.path.join(save_dir, "Y_affine.mat")

    savemat(W_path, {"W_affine": W_mat})
    savemat(Y_path, {"Y_affine": Y_mat})

    print(f"\n✅ Saved W_affine to: {W_path}")
    print(f"✅ Saved Y_affine to: {Y_path}")

    # (Optional) immediately test K evaluation and save K_task as well
    x_eval = np.random.uniform(-1, 1, size=(len(W['Wj']),))
    K_full, K_task = compute_K_task(x_eval, W, Y, task_idx=2)

    print("-----------------------------------------------------------------------------------------")
    print(K_task)
    savemat(os.path.join(save_dir, "K_task_example.mat"),
            {"K_full": K_full, "K_task": K_task, "x_eval": x_eval})
    print("✅ Saved example K_task to: saved_ccm/K_task_example.mat")
else:
    print("⚠️ CCM solution not found. Nothing to save.")


