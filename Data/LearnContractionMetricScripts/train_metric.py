import argparse
import os
from datetime import datetime



import sys
sys.path.append("..") # Adds higher directory to python modules path.



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

from torch.utils.tensorboard import SummaryWriter
import os


log_dir = "./runs/contraction_metric"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

BATCH = 1024
# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 8
batch_size = 512
num_epochs = 100
learning_rate = 1e-3
task_dim = 2  # index of force state
integral = None

json_path_model  = "C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/" \
"experimentsdata_20k_normalizing_constants.json"

checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/models_env_trained/model_new_data_20K_1/model_380.pth'

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"✅ Loaded checkpoint from '{path}' at epoch {checkpoint['epoch']}")
    return checkpoint

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BB_Lyap_Mean(hidden_size=64, ts = 0.001, json_file_path=json_path_model).to(device)
load_checkpoint(checkpoint_path, model)
model.eval()

# Dataset and loader

ts = 0.001

dataset = HydraulicDataset(
    csv_file='C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/train_data_metric_new_model_lambda_2.csv'
    #csv_file='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalized_with_last_x_feature.csv'
)

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
M_net = MetricNet(state_dim).to(device)
optimizer = torch.optim.Adam(M_net.parameters(), lr=learning_rate)

json_path = "C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/train_model_data/data/new_data/" \
"experimentsdata_20k_normalizing_constants.json"

with open(json_path) as f:
    normalizing_constants = json.load(f)

last_desired_force = 0
last_action = 0

import torch

def fd_jacobian(func, x, eps=1e-4, mode='forward'):
    """
    Generic finite-difference Jacobian (batched, vectorized).

    Args:
        func: callable mapping (B, n) -> (B, m)
        x:    (B, n) tensor (the variable we differentiate w.r.t.)
        eps:  finite-difference step
        mode: 'forward' or 'central'

    Returns:
        J: (B, m, n) Jacobian d func / d x
    """
    B, n = x.shape
    x = x.contiguous()
    f0 = func(x)                          # (B, m)
    m = f0.shape[1]
    eye = torch.eye(n, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, -1, -1)  # (B,n,n)

    if mode == 'forward':
        # Build all perturbed inputs at once: (B, n, n) -> (B*n, n)
        x_plus = x.unsqueeze(1) + eps * eye             # (B, n, n)
        x_plus = x_plus.reshape(B * n, n)
        y_plus = func(x_plus).reshape(B, n, m)          # (B, n, m)
        # J[:, :, i] = (f(x+eps e_i) - f(x)) / eps
        J = (y_plus - f0.unsqueeze(1)) / eps            # (B, n, m)
        return J.transpose(1, 2).contiguous()           # (B, m, n)

    elif mode == 'central':
        x_plus  = (x.unsqueeze(1) + eps * eye).reshape(B * n, n)   # (B*n, n)
        x_minus = (x.unsqueeze(1) - eps * eye).reshape(B * n, n)
        y_plus  = func(x_plus).reshape(B, n, m)                    # (B, n, m)
        y_minus = func(x_minus).reshape(B, n, m)
        J = (y_plus - y_minus) / (2.0 * eps)                       # (B, n, m)
        return J.transpose(1, 2).contiguous()                      # (B, m, n)
    else:
        raise ValueError("mode must be 'forward' or 'central'")

def _controller_single(x,ref,integral_error,last_desired_force,u,Kp,Ki,stable):
    global normalizing_constants

    ps = 16000000
    pt = 0
    vpl = 0.00121

    #vpl = model_analytic.Vpl  # Use the model's Vpl parameter
    alpha = 0.601
    Ap = 2.01e-4

    pn = 7000000             
    qn = 10/60000             
    In = 0.050  
    L = 0.08 
    Be = 1.34e9
    timestep_duration = ts
     
    Kv = qn/(In*(pn/2)**(0.5));  


    keys = ['F_actuator', 'F_actuator_deriv', 'F_load', 'F_load_deriv',
                'Pa', 'Pb', 'x', 'x_deriv']  # last_x shares x scale
    
    get_norm = lambda key: (
            normalizing_constants[key]['min'],
            normalizing_constants[key]['max']
        )
        
    mins = [get_norm(k)[0] for k in keys]
    maxs = [get_norm(k)[1] for k in keys]


    x_denorm = x * torch.tensor([maxs[i] - mins[i] for i in range(8)], device=x.device) + \
                torch.tensor(mins, device=x.device)
    
    u_denorm = u*0.1 - 0.05


    (F_actuator, F_actuator_deriv, F_load, F_load_deriv,
        Pa, Pb, x_pos, x_deriv) = [x_denorm[j] for j in range(8)]
    
    F_actuator = F_actuator.item()
    F_load = F_load.item()
    Pa = Pa.item()  
    Pb = Pb.item()
    x_pos = x_pos.item()
    x_deriv = x_deriv.item()
    F_load_deriv = F_load_deriv.item()
    F_actuator_deriv = F_actuator_deriv.item()

    F_load_norm = x.squeeze()[2].item()
    min_force, max_force = get_norm('F_load')
    F_load = F_load_norm * (max_force - min_force) + min_force

    desired_force_norm = ref
    desired_force = desired_force_norm*(max_force - min_force) + min_force
    min_force, max_force = get_norm('F_load')

    min_force, max_force = get_norm('F_load')
    F_load = F_load_norm * (max_force - min_force) + min_force

    force_error = desired_force - F_load
    error = force_error
    integral_error = integral_error + error*timestep_duration



    va = Ap*x_pos + vpl
    vb = alpha*Ap*(L - x_pos) + vpl

    h = -Be*Ap*Ap*(alpha*alpha/vb + 1/va)*x_deriv

    dref = (desired_force - last_desired_force)/timestep_duration

    g1 = Be*Ap*Kv*(((abs(ps - Pa))**(0.5))/va + alpha*((abs(Pb - pt))**(0.5))/vb)
    g2 = Be*Ap*Kv*(((abs(Pa- pt))**(0.5))/va + alpha*((abs(ps - Pb))**(0.5))/vb)

    if(u_denorm >= 0):
        g = g1
    else:
        g = g2


    max_d = 0.0005
    min_d = -0.0005
    mean_d = 0.000

    max_c1 = 1.1
    min_c1 = 0.9
    mean_c1 = 1.0

    max_c2 = 1.1
    min_c2 = 0.9
    mean_c2 = np.sqrt(max_c1*min_c1)

    eta = 1
    beta = np.sqrt(max_c2/min_c2)
    psi = 600

    kp = Kp
    ki = Ki

    s = -kp*error -ki*integral_error

    dkp = 0
    dki = 0

    u = kp*(-mean_c1*h - g*mean_d + dref) + dkp*error + ki*error + dki*integral_error
    k = np.abs(beta - 1)*np.abs(u) + beta*kp*(max_c1 - mean_c1)*np.abs(h) + beta*kp*(max_d - mean_d)*np.abs(g) + eta*beta
    
    sat_value = 0

    if np.abs(s) >= psi:
        sat_value = np.sign(s)
    else:
        sat_value = s/psi
    

    action = (u - k*sat_value)/(mean_c2*kp*g)

    action = (-h + dref + Kp*error + Ki*integral_error)/g

    if(stable == 0):
        action = (Kp*error + Ki*integral_error)/g

    action = np.clip(action,-0.05,0.05)
    action = action/0.05
    last_action = action
    action = action/2 + 0.5

    last_desired_force = desired_force

    return action


def controller(x_batch, ref_batch,stable,integral,last_ref,u,Kp,Ki):
    """
    Vectorized controller over a batch of (B, 8) states and (B,) refs.
    Returns: (B, 1) tensor of actions
    """
    actions = []

    for i in range(x_batch.shape[0]):
        xi = x_batch[i]
        ri = ref_batch[i].item()
        stable_ = stable[i].item()
        integral_= integral[i].item()
        last_ref_= last_ref[i].item()
        u_ = u[i].item()
        Kp_= Kp[i].item()
        Ki_= Ki[i].item()

        if(stable_ >= 0.5):
            action = _controller_single(xi, ri, integral_, last_ref_, u_, Kp_, Ki_,stable_ >= 0.5)
        else:
            action = _controller_single(xi, ri, integral_, last_ref_, u_, Kp_, -Ki_,stable_ >= 0.5)

        actions.append(action)
    return torch.tensor(actions, device=x_batch.device).unsqueeze(1).float()  # (B, 1)

def symmetrize(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))

def finite_difference_jacobian_vectorized(f, x, delta=1e-4):
    """
    Compute Jacobian df/dx using vectorized finite differences.
    f: function mapping (B, n) -> (B, m)
    x: tensor of shape (B, n)
    Returns: (B, m, n) Jacobian tensor
    """
    B, n = x.shape
    device = x.device

    # Create perturbations for all dimensions
    eye = torch.eye(n, device=device).unsqueeze(0)  # (1, n, n)
    eye = eye.repeat(B, 1, 1)                       # (B, n, n)

    # Perturbed x for all dims (forward and backward)
    x_expanded = x.unsqueeze(1)                     # (B, 1, n)
    x_pert_plus  = x_expanded + delta * eye         # (B, n, n)
    x_pert_minus = x_expanded - delta * eye         # (B, n, n)

    # Merge batch and perturb dims to run in one forward call
    x_pert_plus  = x_pert_plus.reshape(B * n, n)
    x_pert_minus = x_pert_minus.reshape(B * n, n)

    # Forward passes
    f_plus  = f(x_pert_plus)   # (B*n, m)
    f_minus = f(x_pert_minus)  # (B*n, m)

    # Reshape back to (B, n, m)
    f_plus  = f_plus.reshape(B, n, -1)
    f_minus = f_minus.reshape(B, n, -1)

    # Central difference
    jac = (f_plus - f_minus) / (2 * delta)  # (B, n, m)

    # Transpose to (B, m, n)
    return jac.transpose(1, 2)


def finite_difference_jacobian(func, x, delta=1e-4):
    B, n = x.shape
    fx = func(x)
    m = fx.shape[1]
    J = torch.zeros(B, m, n, device=x.device, dtype=x.dtype)
    for i in range(n):
        dx = torch.zeros_like(x)
        dx[:, i] += delta
        fx_perturbed = func(x + dx)
        J[:, :, i] = (fx_perturbed - fx) / delta
    return J

def finite_difference_time_derivative(M_net, x, f, delta_t=ts):
    x_next = x + f * delta_t
    M_now = M_net(x)
    M_next = M_net(x_next)
    return (M_next - M_now) / delta_t

def contraction_certificate_value_Horizon(
    x, u, ref,integral,last_ref,f_model, M_net, policy,
    is_stable_mask,KP,KI,
    delta=1e-4,
    lambda_val=0.1,
    task_dim=2,
    reg_weight=1e-3
):
    """
    Full contraction loss with contrastive logic:
    - Penalize contraction violations on stable data.
    - Penalize contraction satisfaction on unstable data.
    """
    is_stable = is_stable_mask > 0.5
    is_unstable = ~is_stable

    B, n = x.shape
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    def model(xu): 
        a = f_model(2*xu - 1)
        a = a/2 + 0.5
        return a

    f_x_fn = lambda x_: model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref,is_stable,integral,last_ref,u,KP,KI)

    A = finite_difference_jacobian(f_x_fn, x, delta=delta)
    B_mat = finite_difference_jacobian(f_u_fn, u, delta=delta)
    K = finite_difference_jacobian(policy_fn, x, delta=delta)
    f_val = model(torch.cat([x, u], dim=-1))

    M = M_net(x)
    M_dot = finite_difference_time_derivative(M_net, x, f_val)

    v = torch.zeros(B, n, 1, device=x.device)
    v[:, task_dim, 0] = x[:, task_dim] - ref

    BK = torch.bmm(B_mat, K)
    A_plus_BK = A + BK
    sym_term = symmetrize(torch.bmm(M, A_plus_BK))

    contraction_matrix = M_dot + sym_term + 2 * lambda_val * M
    vT_C = torch.bmm(v.transpose(1, 2), torch.bmm(contraction_matrix, v))
    cert_val = vT_C.squeeze(-1).squeeze(-1)

    return cert_val, {
        "M": M,
        "M_dot": M_dot,
        "A": A,
        "B": B_mat,
        "K": K,
        "BK": BK,
        "sym_term": sym_term,
        "contraction_matrix": contraction_matrix,
        "v": v
    }

def contraction_certificate_value(
    x, u, ref, f_model, M_net, policy,stable,
    delta=1e-4, lambda_val=1, task_dim=2
):
    """
    Full contraction condition with respect to the force tracking error direction.

    Computes: vᵀ (Ṁ + sym(M(A + BK)) + 2λM) v

    Args:
        x:         (B, n) normalized states
        u:         (B, m) control input
        ref:       (B,) reference force (normalized scalar)
        f_model:   dynamics model, maps (B, n + m) -> (B, n)
        M_net:     metric model, maps (B, n) -> (B, n, n)
        policy:    control function, maps (B, n), (B,) -> (B, m)
        delta:     finite difference epsilon
        lambda_val: contraction rate λ
        task_dim:  index of force-tracked state

    Returns:
        cert_val: (B,) contraction condition values
        debug: dict of all intermediate tensors
    """
    B, n = x.shape
    m = u.shape[1]

    # Dynamics
    f_x_fn = lambda x_: f_model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: f_model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref, stable)

    A = finite_difference_jacobian(f_x_fn, x, delta=delta)      # (B, n, n)
    B_mat = finite_difference_jacobian(f_u_fn, u, delta=delta)  # (B, n, m)
    K = finite_difference_jacobian(policy_fn, x, delta=delta)   # (B, m, n)

    # Compute f(x, u)
    f_val = f_model(torch.cat([x, u], dim=-1))  # (B, n)

    # M and Ṁ
    M = M_net(x)                                # (B, n, n)
    M_dot = finite_difference_time_derivative(M_net, x, f_val)  # (B, n, n)

    # v = (x - x_ref) direction, only on task_dim
    v = torch.zeros(B, n, 1, device=x.device)
    v[:, task_dim, 0] = x[:, task_dim] - ref    # tracking direction

    # Compute M(A + BK)
    BK = torch.bmm(B_mat, K)                    # (B, n, n)
    A_plus_BK = A + BK                          # (B, n, n)
    sym_term = symmetrize(torch.bmm(M, A_plus_BK))  # (B, n, n)

    contraction_matrix = M_dot + sym_term + 2 * lambda_val * M  # (B, n, n)

    # Contraction value vᵀ C v
    vT_C = torch.bmm(v.transpose(1, 2), torch.bmm(contraction_matrix, v))  # (B, 1, 1)
    cert_val = vT_C.squeeze(-1).squeeze(-1)  # (B,)

    return cert_val, {
        "M": M,
        "M_dot": M_dot,
        "A": A,
        "B": B_mat,
        "K": K,
        "BK": BK,
        "sym_term": sym_term,
        "contraction_matrix": contraction_matrix,
        "v": v
    }

def contraction_tracking_loss_with_contrast(
    x, u, ref, f_model, M_net, policy,
    is_stable_mask,
    delta=1e-4,
    lambda_val=1,
    task_dim=2,
    reg_weight=1e-3
):
    """
    Full contraction loss with contrastive logic:
    - Penalize contraction violations on stable data.
    - Penalize contraction satisfaction on unstable data.
    """
    is_stable = is_stable_mask > 0.5
    is_unstable = ~is_stable

    B, n = x.shape
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    f_x_fn = lambda x_: f_model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: f_model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref,is_stable)

    A = finite_difference_jacobian(f_x_fn, x, delta=delta)
    B_mat = finite_difference_jacobian(f_u_fn, u, delta=delta)
    K = finite_difference_jacobian(policy_fn, x, delta=delta)
    f_val = f_model(torch.cat([x, u], dim=-1))

    M = M_net(x)
    M_dot = finite_difference_time_derivative(M_net, x, f_val)

    v = torch.zeros(B, n, 1, device=x.device)
    v[:, task_dim, 0] = x[:, task_dim] - ref

    BK = torch.bmm(B_mat, K)
    A_plus_BK = A + BK
    sym_term = symmetrize(torch.bmm(M, A_plus_BK))

    contraction_matrix = M_dot + sym_term + 2 * lambda_val * M
    vT_C = torch.bmm(v.transpose(1, 2), torch.bmm(contraction_matrix, v))
    cert_val = vT_C.squeeze(-1).squeeze(-1)



    # (B,)

    # STABLE → penalize cert_val > 0
    stable_loss = torch.relu(cert_val[is_stable]).mean() if is_stable.any() else 0.0

    # UNSTABLE → penalize cert_val < 0 (i.e., reward violation of contraction)
    unstable_loss = torch.relu(-cert_val[is_unstable]).mean() if is_unstable.any() else 0.0

    contraction_loss_weight = 40000000
    total_loss = contraction_loss_weight*(stable_loss + unstable_loss)

    contraction_loss = total_loss

    fro_norm = torch.mean(torch.linalg.norm(M, dim=(1, 2)))           # Frobenius norm
    trace_reg = M.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1).mean()                           # Trace (positive sum of eigenvalues)
    logdet_reg = -torch.mean(torch.logdet(M + 1e-6 * torch.eye(M.shape[-1], device=M.device)))  # Penalize small det

    # Weight terms
    reg_weight_fro = 1e-3
    reg_weight_trace = 1e-3
    reg_weight_logdet = 1e-3

    regularization = (
        reg_weight_fro * fro_norm +
        reg_weight_trace * trace_reg +
        reg_weight_logdet * logdet_reg
    )

    total_loss = contraction_loss + regularization
    

    return total_loss, contraction_loss.item(), regularization.item()

def compute_contract_loss(
    x, u, ref,integral,last_ref,f_model, M_net, policy,
    is_stable_mask,Kp,Ki,
    delta=1e-4,
    lambda_val=0.1,
    task_dim=2,
    reg_weight=1e-3
):
    """
    Full contraction loss with contrastive logic:
    - Penalize contraction violations on stable data.
    - Penalize contraction satisfaction on unstable data.
    """
    is_stable = is_stable_mask > 0.5
    is_unstable = ~is_stable

    B, n = x.shape
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    def model(xu): 
        a = f_model(2*xu - 1)
        a = a/2 + 0.5
        return a


    f_x_fn = lambda x_: model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref,is_stable, integral,last_ref,u,Kp,Ki)

    A = finite_difference_jacobian(f_x_fn, x, delta=delta)
    B_mat = finite_difference_jacobian(f_u_fn, u, delta=delta)
    K = finite_difference_jacobian(policy_fn, x, delta=delta)
    f_val = model(torch.cat([x, u], dim=-1))

    M = M_net(x)
    M_dot = finite_difference_time_derivative(M_net, x, f_val)

    v = torch.zeros(B, n, 1, device=x.device)
    v[:, task_dim, 0] = x[:, task_dim] - ref

    BK = torch.bmm(B_mat, K)
    A_plus_BK = A + BK
    sym_term = symmetrize(torch.bmm(M, A_plus_BK))

    lambda_val = lambda_val.to(M.device, dtype=M.dtype)

    # If lambda_val is scalar: (e.g., a single float)
    if lambda_val.dim() == 0:
        scaled_M = lambda_val * M                    # (B,8,8)

    # If lambda_val is per-sample: shape (B,) or (B,1)
    elif lambda_val.dim() == 1:                      # (B,)
        scaled_M = lambda_val.view(-1, 1, 1) * M     # (B,8,8)
    elif lambda_val.dim() == 2 and lambda_val.size(1) == 1:  # (B,1)
        scaled_M = lambda_val.view(-1, 1, 1) * M     # (B,8,8)
    else:
        raise ValueError(f"Unexpected lambda_val shape: {tuple(lambda_val.shape)}")

    contraction_matrix = M_dot + sym_term + 2 * scaled_M
    vT_C = torch.bmm(v.transpose(1, 2), torch.bmm(contraction_matrix, v))
    cert_val = vT_C.squeeze(-1).squeeze(-1)

    # (B,)

    # STABLE → penalize cert_val > 0
    margin = 1e-3
    stable_loss = torch.relu(cert_val[is_stable] + margin).mean() if is_stable.any() else 0.0

    # UNSTABLE → penalize cert_val < 0 (i.e., reward violation of contraction)
    unstable_loss = torch.relu(-cert_val[is_unstable]).mean() if is_unstable.any() else 0.0

    contraction_loss_weight = 100
    total_loss = contraction_loss_weight*(stable_loss + unstable_loss)

    contraction_loss = total_loss

    fro_norm = torch.mean(torch.linalg.norm(M, dim=(1, 2)))           # Frobenius norm
    trace_reg = M.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1).mean()                           # Trace (positive sum of eigenvalues)
    logdet_reg = -torch.mean(torch.logdet(M + 1e-6 * torch.eye(M.shape[-1], device=M.device)))  # Penalize small det

    # Weight terms
    reg_weight_fro = 1e-3
    reg_weight_trace = 1e-3
    reg_weight_logdet = 1e-3

    regularization = (
        reg_weight_fro * fro_norm +
        reg_weight_trace * trace_reg +
        reg_weight_logdet * logdet_reg
    )

    total_loss = contraction_loss + regularization
    

    return contraction_loss, regularization

def contraction_tracking_loss_continuous(
    x, x_next, ref, f_model, M_net, policy, is_stable_mask,
    delta=1e-4, lambda_val=1.0, task_dim=2, reg_weight=1e-3
):
    """
    Full continuous-time contraction loss using finite differences and contrastive penalties.

    Args:
        x: (B, n) current state
        x_next: (B, n) next state from dataset
        ref: (B,) normalized scalar reference
        f_model: dynamics model, maps (x, u) -> x_next
        M_net: contraction metric network, maps x -> (n x n) PSD matrix
        policy: control function u = π(x, ref)
        is_stable_mask: (B,) boolean tensor
        delta: finite-diff step
        lambda_val: contraction rate
        task_dim: index of force to track
        reg_weight: regularization coefficient

    Returns:
        total_loss, contraction_loss, regularization_loss
    """
    is_stable = is_stable_mask > 0.5
    is_unstable = ~is_stable

    B, n = x.shape
    u = policy(x, ref,is_stable)

    # Define finite-diff functions
    f_x_fn = lambda x_: f_model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: f_model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref, is_stable)

    # Estimate derivatives
    A = finite_difference_jacobian(f_x_fn, x, delta=delta)      # (B, n, n)
    B_mat = finite_difference_jacobian(f_u_fn, u, delta=delta)  # (B, n, m)
    K = finite_difference_jacobian(policy_fn, x, delta=delta)   # (B, m, n)

    # Estimate f(x,u) from dataset as f = x_next - x (discrete to continuous)
    f_val = x_next - x  # (B, n) as approximation of dx/dt

    # Evaluate M(x) and Ṁ
    M = M_net(x)  # (B, n, n)
    M_dot = finite_difference_time_derivative(M_net, x, f_val)  # (B, n, n)

    # Compute BK, A+BK, and symmetric term
    BK = torch.bmm(B_mat, K)                # (B, n, n)
    A_plus_BK = A + BK                      # (B, n, n)
    sym_term = symmetrize(torch.bmm(M, A_plus_BK))  # (B, n, n)

    # Contraction matrix: Ṁ + sym(M(A+BK)) + 2λM
    contraction_matrix = M_dot + sym_term + 2 * lambda_val * M  # (B, n, n)

    # Direction vector v (tracking error in force direction)
    v = torch.zeros(B, n, 1, device=x.device)
    v[:, task_dim, 0] = x[:, task_dim] - ref

    # Certificate: vᵀ C v
    cert = torch.bmm(v.transpose(1, 2), torch.bmm(contraction_matrix, v)).squeeze(-1).squeeze(-1)  # (B,)

    # STABLE: Penalize cert > 0
    loss_stable = torch.relu(cert[is_stable]).mean() if is_stable.any() else 0.0

    # UNSTABLE: Penalize cert < 0
    loss_unstable = torch.relu(-cert[is_unstable]).mean() if is_unstable.any() else 0.0

    # Regularization terms for metric
    fro_norm = torch.mean(torch.linalg.norm(M, dim=(1, 2)))           # Frobenius norm
    trace_reg = M.diagonal(offset=0, dim1=1, dim2=2).sum(dim=1).mean()
    logdet_reg = -torch.mean(torch.logdet(M + 1e-6 * torch.eye(M.shape[-1], device=M.device)))  # Keep det > 0

    reg_loss = (
        reg_weight * fro_norm +
        reg_weight * trace_reg +
        reg_weight * logdet_reg
    )

    contraction_loss_weight = 40000000

    total_loss = contraction_loss_weight*(loss_stable + loss_unstable)
    
    contraction_loss = total_loss

    total_loss = contraction_loss + reg_loss

    return total_loss, contraction_loss.item(), reg_loss.item()

def contraction_tracking_loss_multistep(
    x_batch,
    f_model, M_net, policy, is_stable,T, task_dim=None
):
    """
    Contraction tracking loss using measured multi-step trajectory segments.

    Args:
        x           (B, n)       : current state
        u           (B, m)       : current control
        ref         (B, r)       : current reference
        x_futures   (B, T, n)    : future states from dataset
        u_futures   (B, T, m)    : future controls from dataset
        ref_futures (B, T, r)    : future references from dataset
        M_net       (nn.Module)  : metric network
        policy      (callable)   : policy network (unused if using measured u)
        is_stable   (B,)         : stability labels
    """
    a_ = 15
    contrac_loss_total = 0
    reg_loss_total = 0 
    
    for step in range(T):
        x = x_batch[:, ((step)*a_ + 0):(step*a_ + 8)]      # state: first 8 dims
        u = x_batch[:, (step*a_ + 10)]                  
        ref = x_batch[:, (step*a_ + 11)]                       # ref: 10th column (index 9)
        integral = x_batch[:, (step*a_ + 8)]                       # ref: 10th column (index 9)
        last_ref = x_batch[:, (step*a_ + 9)]                       # ref: 10th column (index 9)
        Kp = x_batch[:, (step*a_ + 12)]
        Ki = x_batch[:, (step*a_ + 13)]
        lambda_value = x_batch[:, (step*a_ + 14)]

        #print(x.shape)
        #print(u.shape)
        #print(ref.shape)
        #print(integral.shape)
        #print(last_ref.shape)

        contrac_loss,reg_loss = compute_contract_loss(x,u,ref,integral,last_ref,f_model, M_net, policy, is_stable,Kp,Ki, lambda_val=lambda_value, task_dim=task_dim)

        contrac_loss_total +=contrac_loss
        reg_loss_total += reg_loss
    # Average across steps

    contrac_loss_total = contrac_loss_total/T
    reg_loss_total = reg_loss_total/T

    total_loss = contrac_loss_total + reg_loss_total
 
    return total_loss, contrac_loss_total.item(), reg_loss_total.item()

def get_device(preferred_gpu=1):
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        if preferred_gpu < num_gpus:
            return torch.device(f"cuda:{preferred_gpu}")
        else:
            print(f"Warning: Only {num_gpus} GPU(s) available. Falling back to cuda:0.")
            return torch.device("cuda:0")
    return torch.device("cpu")

def load_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"✅ Loaded checkpoint from '{path}' at epoch {start_epoch}")
    return start_epoch

def save_checkpoint(model, optimizer, epoch, loss, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"model_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)

running_loss = 0.0

def train(model_M, optimizer, criterion, dataloader, writer, log_dir, num_epochs, start_epoch=0,scheduller = None):
    global model
    global running_loss
    device = get_device()
    model.to(device)
    model_M.to(device)

    for epoch in range(num_epochs):
        # Wrap dataloader in tqdm for progress bar
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
        

        for x_batch, y_batch in loop:
            x_batch = x_batch.to(device).float()
            is_stable = x_batch[:, -1]                # Stable: last column


            loss, loss_contr, loss_reg = contraction_tracking_loss_multistep(
                x_batch, model.single_step, model_M, controller, is_stable,T = 5,
                task_dim=task_dim
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item(), contr=loss_contr, reg=loss_reg)

            # TensorBoard logging
            step = epoch * len(dataloader) + loop.n
            writer.add_scalar("Loss/Total", loss.item(), step)
            writer.add_scalar("Loss/Contraction", loss_contr, step)
            writer.add_scalar("Loss/Regularization", loss_reg, step)

            if(step % 50 == 0):
                save_checkpoint(model_M, optimizer, step, 0, log_dir)

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        print(f"Epoch [{epoch + 1}/{num_epochs}] → Loss: {avg_loss:.6f}")
        scheduller.step(avg_loss)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)


        #save_checkpoint(model_M, optimizer, epoch, avg_loss, log_dir)


def main():
    parser = argparse.ArgumentParser(description="Train MLP with optional checkpoint resume")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a .pth checkpoint to resume from")
    args = parser.parse_args()

    device = get_device()

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    model_M = MetricNet(state_dim=state_dim,
        hidden_dim=32  # Increased from 64
        #json_file_path='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalizing_constants.json'
    )

    criterion = nn.MSELoss()

    optimizer = optim.Adam(model_M.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,    # Reduce by half (gentler than 0.1, which is aggressive)
        patience=8    # Faster reaction but not too sensitive
    )


    log_dir = os.path.join("runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    writer = SummaryWriter(log_dir)

    start_epoch = 0
    if args.checkpoint:
        start_epoch = load_checkpoint(args.checkpoint, model_M, optimizer)

    global integral
    integral  = np.zeros((BATCH, 1))

    train(
        model_M=model_M,
        optimizer=optimizer,
        criterion=criterion,
        dataloader=dataloader,
        writer=writer,
        log_dir=log_dir,
        num_epochs=5,  
        start_epoch=start_epoch,
        scheduller=scheduler
    )

    writer.close()


if __name__ == "__main__":
    main()
