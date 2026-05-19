import torch
from torch.utils.data import DataLoader
from models_lyap.dataset_lyap import HydraulicDataset
from models_lyap.MetricNet import MetricNet
import numpy as np
import json
from models.run_bb_mean_lyap_4.mlp_lyap_mean_bb import BB_Lyap_Mean

from torch.utils.tensorboard import SummaryWriter
import os

log_dir = "./runs/contraction_metric"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)


# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
state_dim = 8
batch_size = 512
num_epochs = 100
learning_rate = 1e-3
task_dim = 2  # index of force state

json_path_model  = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/data_norm_normalizing_constants.json'
checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/run_bb_mean_lyap_4/model_999.pth'

def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"✅ Loaded checkpoint from '{path}' at epoch {checkpoint['epoch']}")
    return checkpoint

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = BB_Lyap_Mean(hidden_size=128, json_file_path=json_path_model).to(device)
load_checkpoint(checkpoint_path, model)
model.eval()

# Dataset and loader

dataset = HydraulicDataset(
    csv_file='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/stable_trajectory_test_prep.csv'
    #csv_file='/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/raw_data/0_2025-05-19-senoidal-300A-0.5to3hz/0_2025-05-19-senoidal-300A-0.5to3hz_normalized_with_last_x_feature.csv'
)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model and optimizer
M_net = MetricNet(state_dim).to(device)
optimizer = torch.optim.Adam(M_net.parameters(), lr=learning_rate)

json_path = "/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models_lyap/new_constants.json"
with open(json_path) as f:
    normalizing_constants = json.load(f)

last_desired_force = 0
last_action = 0
integral_error=0

def _controller_single(x,ref):
    global normalizing_constants
    global last_desired_force
    global last_action
    global integral_error

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
    timestep_duration = 0.001
     
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

    if(last_action >= 0):
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

    kp = 15
    ki = 5

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

    action = (-h + dref - 30*error - 15*integral_error)/g

    action = np.clip(action,-0.05,0.05)
    action = action/0.05
    last_action = action
    action = action/2 + 0.5

    last_desired_force = desired_force

    return action

def controller(x_batch, ref_batch):
    """
    Vectorized controller over a batch of (B, 8) states and (B,) refs.
    Returns: (B, 1) tensor of actions
    """
    actions = []
    for i in range(x_batch.shape[0]):
        xi = x_batch[i]
        ri = ref_batch[i].item()
        action = _controller_single(xi, ri)
        actions.append(action)
    return torch.tensor(actions, device=x_batch.device).unsqueeze(1).float()  # (B, 1)

import torch
import torch

def symmetrize(matrix):
    return 0.5 * (matrix + matrix.transpose(-1, -2))

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

def finite_difference_time_derivative(M_net, x, f, delta_t=1e-3):
    x_next = x + f * delta_t
    M_now = M_net(x)
    M_next = M_net(x_next)
    return (M_next - M_now) / delta_t

def contraction_certificate_value(
    x, u, ref, f_model, M_net, policy,
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
    policy_fn = lambda x_: policy(x_, ref)

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

def contraction_tracking_loss_new(
    x, u, ref, f_model, M_net, policy,
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
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    # Dynamics
    f_x_fn = lambda x_: f_model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: f_model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref)

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

    loss = torch.relu(cert_val)  # relu(x) = max(0, x)
    return loss.mean()  # mean loss over batch

def contraction_tracking_loss_discrete(
    x,
    ref_force_scalar,
    dynamics_model,
    policy,
    M_net,
    is_stable_mask,
    task_dim=2,
    delta=1e-4
):
    """
    Discrete-time contraction loss for force tracking along one state dimension.

    Args:
        x:                (B, n) current normalized states
        ref_force_scalar: scalar in [0, 1], normalized desired force for all samples
        dynamics_model:   model predicting x_{t+1} = f(x, u)
        policy:           controller function u = π(x)
        M_net:            contraction metric M(x)
        is_stable_mask:   (B,) boolean mask for stable trajectories
        task_dim:         index of the force state to track (e.g., 2)
        delta:            contraction margin

    Returns:
        loss: scalar
        contraction_val: (B,) contraction violation values
        e_now: (B,) v_tᵀ M(x_t) v_t
        e_next: (B,) v_{t+1}ᵀ M(x_{t+1}) v_{t+1}
    """
    B, n = x.shape
    is_stable = is_stable_mask > 0.5

    # Detach and enable gradient tracking for M_net
    x = x.detach().clone().requires_grad_(True)

    # Controller action
    u = policy(x,ref_force_scalar)  # (B, m)

    # Next state prediction
    x_next = dynamics_model(torch.cat([x, u], dim=-1))  # (B, n)
    x_next = x_next.detach().clone().requires_grad_(True)

    # Reference force is scalar, broadcast to match batch
    ref_force = ref_force_scalar

    # Tracking directions: v_t = x[:, task_dim] - ref
    v_t = torch.zeros_like(x)
    v_t[:, task_dim] = x[:, task_dim] - ref_force
    v_t = v_t.view(B, n, 1)

    v_tp1 = torch.zeros_like(x_next)
    v_tp1[:, task_dim] = x_next[:, task_dim] - ref_force
    v_tp1 = v_tp1.view(B, n, 1)

    # Metric
    M_now = M_net(x)         # (B, n, n)
    M_next = M_net(x_next)   # (B, n, n)

    # Energy terms
    e_now = torch.bmm(v_t.transpose(1, 2), torch.bmm(M_now, v_t)).squeeze()       # (B,)
    e_next = torch.bmm(v_tp1.transpose(1, 2), torch.bmm(M_next, v_tp1)).squeeze() # (B,)

    contraction_val = e_next - e_now + delta

    loss = torch.relu(contraction_val[is_stable]).mean() if is_stable.any() else torch.tensor(0.0, device=x.device)

    return loss, contraction_val.detach(), e_now.detach(), e_next.detach()

def contraction_tracking_loss_new_with_reg(
    x, u, ref, f_model, M_net, policy,
    delta=1e-4, lambda_val=1, task_dim=2, reg_weight=1e-3
):
    B, n = x.shape
    if u.ndim == 1:
        u = u.unsqueeze(-1)

    f_x_fn = lambda x_: f_model(torch.cat([x_, u], dim=-1))
    f_u_fn = lambda u_: f_model(torch.cat([x, u_], dim=-1))
    policy_fn = lambda x_: policy(x_, ref)

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

    # Primary loss
    contraction_loss = torch.relu(cert_val).mean()

    # Regularization: Frobenius norm or trace of M
    reg = torch.mean(torch.linalg.norm(M, dim=(1, 2)))  # Frobenius norm
    # alt: reg = torch.mean(torch.trace(M))  # use only if you want trace

    total_loss = contraction_loss + reg_weight * reg
    return total_loss, contraction_loss.item(), reg.item()




from tqdm import tqdm

for epoch in range(num_epochs):
    # Wrap dataloader in tqdm for progress bar
    loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

    for x_batch, y_batch in loop:
        x_batch = x_batch.to(device).float()
        x = x_batch[:, :8]      # state: first 8 dims
        u = x_batch[:, 8]                  
        ref = x_batch[:, 9]                       # ref: 10th column (index 9)
        is_stable = x_batch[:, -1]                # Stable: last column

        loss, loss_contr, loss_reg = contraction_tracking_loss_new_with_reg(
            x, u, ref, model.single_step, M_net, controller,
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

        # Update tqdm postfix with current loss
        loop.set_postfix(loss=loss.item())

        torch.save({
            'epoch': epoch,
            'model_state_dict': M_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, f"/checkpoints/contraction_epoch_{epoch:03d}.pth")


    # Optionally print epoch summary outside tqdm
    print(f"Epoch {epoch+1:03d} completed, last batch loss: {loss.item():.6f}")


