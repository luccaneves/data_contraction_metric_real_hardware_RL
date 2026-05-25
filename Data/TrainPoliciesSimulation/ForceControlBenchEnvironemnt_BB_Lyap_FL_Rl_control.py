import numpy as np
import torch
import pandas as pd
import gymnasium
from gymnasium import spaces
import random
import json
import torch.nn as nn
import cvxpy as cp
from models.run_bb_mean_lyap_4.mlp_lyap_mean_bb import BB_Lyap_Mean
from models_lyap.run_lyap_agora_vai_1.lyap import Lyap


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path, model):
    checkpoint = torch.load(path, map_location=get_device())
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(f"✅ Loaded checkpoint from '{path}' at epoch {checkpoint['epoch']}")
    return checkpoint


class ForceControlBenchEnvironment_BB_Lyap_FL_Rl_control(gymnasium.Env):
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, no_random,only_control,safe_control_strategy,
                 delta = 0, alpha_ = 0, k_sigma = 2.5,Q = 1, R = 50, R2 = 50,
                 action_scale = 0.2, PID_FL_flag = 1, print_flag = False, output_type = 0, param_scale = 1, control_freq = 100, 
                 smooth_factor = 0.5):
        
        # === File paths ===
        json_path = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/data_norm_normalizing_constants.json'
        json_path_lyap = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models_lyap/new_constants.json'
        dataset_csv = 'C:/Users/ic2d/Documents/GitHub/Hyd_Learning/current_branch/data_eval.csv'
        checkpoint_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models/run_bb_mean_lyap_4/model_999.pth'
        checkpoint_lyap_path = '/home/nexus/Documents/GitHub/Hyd_Learning/current_branch/models_lyap/run_lyap1/model_599.pth'

        self.smooth_factor = smooth_factor
        self.no_random = no_random  # Disable randomization for reproducibility
        self.only_control = only_control  # Use control component in action
        self.safe_control_strategy = safe_control_strategy  # Placeholder for face control state
        self.delta = delta  # Delta for Lyapunov control
        self.alpha_ = alpha_  # Alpha for Lyapunov control
        self.k_sigma = k_sigma  # k_sigma for Lyapunov control
        self.action_scale = action_scale
        self.PID_FL_flag = PID_FL_flag  # Flag for PID control in Force Load
        self.print_flag = print_flag  # Flag for printing debug information
        self.output_type = output_type
        self.param_scale = param_scale
        self.control_freq = control_freq
        self.counter = 0

        # === Load data ===
        self.data = pd.read_csv(dataset_csv)
        with open(json_path, 'r') as file:
            self.normalizing_constants = json.load(file)

        self.get_norm = lambda key: (
            self.normalizing_constants[key]['min'],
            self.normalizing_constants[key]['max']
        )

        with open(json_path_lyap, 'r') as file:
            self.normalizing_constants_lyap = json.load(file)

        self.get_norm_lyap = lambda key: (
            self.normalizing_constants_lyap[key]['min'],
            self.normalizing_constants_lyap[key]['max']
        )

        self.device = get_device()

        # === Models ===

        self.model = BB_Lyap_Mean(hidden_size=128, json_file_path=json_path).to(self.device)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()
        #self.model.to(self.device)

        self.model_lyap = Lyap(hidden_size=256, json_file_path=json_path_lyap).to(self.device)
        load_checkpoint(checkpoint_lyap_path, self.model_lyap)
        self.model_lyap.eval()
        #self.model_lyap.to(self.device)
        #torch.set_grad_enabled(True)

        self.ps = 16000000
        self.pt = 0
        self.vpl = torch.tensor([0.00121], dtype=torch.float32, device=self.device)  # Use the model's Vpl parameter
        self.Max_vpl = self.param_scale*self.vpl.item()
        self.Min_vpl = (1/self.param_scale)*self.vpl.item()
        #self.vpl = self.model_analytic.Vpl  # Use the model's Vpl parameter
        self.alpha = 0.601
        self.Ap = 2.01e-4

        self.pn = 7000000             
        self.qn = 10/60000             
        self.In = 0.050  
        self.L = 0.08 
        self.Be = torch.tensor([1.34e9], dtype=torch.float32, device=self.device)  # Use the model's Be parameter
        self.Max_Be = self.param_scale*self.Be.item()
        self.Min_Be = (1/self.param_scale)*self.Be.item()
        #self.Be = self.model_analytic.Be  # Use the model's Be parameter      


        self.Kv = self.qn/(self.In*(self.pn/2)**(0.5));  
        self.Kv = torch.tensor([self.Kv], dtype=torch.float32, device=self.device)  # Use the model's Kv parameter
        self.Max_Kv = self.param_scale*self.Kv.item()
        self.Min_Kv = (1/self.param_scale)*self.Kv.item()
        #self.Kv = self.model_analytic.Kv  # Use the model's Kv parameter 


        self.Max_Kp = 60
        self.Min_Kp = 0
        self.Max_Ki = 30
        self.Min_Ki = 0
        self.Max_Kd = 0
        self.Min_Kd = 0


        # === Observation and Action Spaces ===
        self.obs_size = 12
        self.action_space_sz = 2
        self.observation_space = spaces.Box(
            low=np.array([0] * self.obs_size),
            high=np.array([1] * self.obs_size),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=np.array(self.action_space_sz *[-1]),
            high=np.array(self.action_space_sz *[1]),
            dtype=np.float32
        )

        # === Trajectory setup ===
        self.timestep_duration = 1 / 1000  # 1ms steps
        self.total_timesteps = 2500

        if self.no_random:
            self.sin_freq = 1 * 2 * np.pi
            self.amplitude = 150
        else:
            self.sin_freq = random.uniform(0.5, 3) * 2 * np.pi
            self.amplitude = random.uniform(150, 200)

        # === Cost weights ===
        self.Q = Q
        self.R = R
        self.R2 = R2

        # === State ===
        self.start_state_index = 1000
        self.current_state = None
        self.current_time_step = 0
        self.last_action = 0
        self.last_action_rl = 0

        self._generate_force_trajectory()

    def print(self, print_str):
        if self.print_flag:
            print(print_str)

    def _generate_force_trajectory(self):
        t = np.arange(self.total_timesteps) * self.timestep_duration
        self.Force_desired = (self.amplitude - self.amplitude * np.cos(self.sin_freq * t)).astype(np.float32).reshape(-1, 1)

    def _get_obs(self):
        min_force, max_force = self.get_norm('F_load')
        desired_force = self.Force_desired[self.current_time_step][0]
        desired_force_norm = (desired_force - min_force) / (max_force - min_force)
        

        if self.current_time_step == 0:
            desired_force_norm_deriv = 0
            last_desired_force_norm = desired_force_norm
        else:
            last_desired_force = self.Force_desired[self.current_time_step - 1][0]
            last_desired_force_norm = (last_desired_force - min_force) / (max_force - min_force)
            desired_force_norm_deriv = (desired_force_norm - last_desired_force_norm)


        obs = torch.cat([
            self.current_state.squeeze(),
            torch.tensor([desired_force_norm], dtype=torch.float32, device=self.device),
            torch.tensor([desired_force_norm_deriv], dtype=torch.float32, device=self.device),
            torch.tensor([desired_force_norm - self.current_state.squeeze()[2]], dtype=torch.float32, device=self.device),
            torch.tensor([self.last_action], dtype=torch.float32, device=self.device),
        ])


        return obs.detach().cpu().numpy()

    def _get_info(self):
        return {"info": "Placeholder"}

    def calculate_end(self):
        return self.current_time_step >= (self.total_timesteps - 3)

    def reset(self, seed=None, options=None):
        self.current_time_step = 0
        self.last_action = 0
        self.last_action_rl = 0
        self.V_current = 0
        self.V_next = 0
        self.last_error = 0

        self.integral_error = 0
        self.last_V =  torch.tensor([0], dtype=torch.float32, device=self.device)
        self.last_xt = 0
        self.dV = 0
        self.last_action_filter = 0
        self.grad_u = 0
        self.Vdot_filtered = 0
        self.V = 0
        self.vdot = 0
        self.last_action_qp_filter = 0

        self.smoothed_action = 0

        # Randomize frequency and amplitude
        if self.no_random:
            self.sin_freq = 1 * 2 * np.pi
            self.amplitude = 150
        else:
            self.sin_freq = random.uniform(0.5, 3) * 2 * np.pi
            self.amplitude = random.uniform(150, 200)

        self._generate_force_trajectory()

        init_state = self.data.iloc[self.start_state_index, 1:9].values.astype(np.float32)

        init_state = 8*[0.5]

        self.current_state_bb = torch.tensor([init_state], dtype=torch.float32, device=self.device,requires_grad=True)
        self.current_state_analytic = torch.tensor([init_state], dtype=torch.float32, device=self.device,requires_grad=True)

        self.current_state = torch.cat((
            self.current_state_analytic.squeeze()[0:4],
            self.current_state_bb.squeeze()[4:6],
            self.current_state_analytic.squeeze()[6:8]
        )).unsqueeze(0)

        self.action_control = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def renormalize_tensor(self, x_norm, old_min, old_max, new_min, new_max):
        old_min = torch.tensor(old_min, device=x_norm.device)
        old_max = torch.tensor(old_max, device=x_norm.device)
        new_min = torch.tensor(new_min, device=x_norm.device)
        new_max = torch.tensor(new_max, device=x_norm.device)

        x_denorm = x_norm * (old_max - old_min) + old_min
        return (x_denorm - new_min) / (new_max - new_min)
    
        
    def qp_safe_control(
        self,
        x,                       # current Lyap input [numpy]
        u_nominal,               # torch scalar on device
        u_safe,
        *,                       # keyword‑only params below
        delta_min     = 5e-1,    # absolute floor on −V̇
        delta_gain    = 5e-1,    # α  s.t.  margin = α·|V|   (try 5e‑4 → 5e‑3)
        k_sigma       = 1.0,     # σ multiplier  (1 → 5)
        grad_thresh   = 2e-1,    # skip QP if |dV/du| is weaker
        du_clip       = 0.2,    # |Δu|max (0.1 → 0.3)
        slack_penalty = 1e4      # weight on slack in objective
    ):
        with torch.enable_grad():
            # --------------------------------------------------- 1. Prep
            x_t = torch.tensor(x, dtype=torch.float32,
                            device=self.device,
                            requires_grad=True).unsqueeze(0) #[0,1]
            
            x_t = x_t.clone()
            x_t[0][0][:8] = self.renormalize_tensor(x_t[0][0][:8],
                                    self.mins,
                                    self.maxs,
                                    self.mins_lyap,
                                    self.maxs_lyap)
        

            first_state = x_t

            # make sure u_nominal is differentiable
            if not u_nominal.requires_grad:
                u_nominal = u_nominal.detach().clone().requires_grad_(True)
            u_flat = u_nominal.view(-1)

            ##############################################################
            # --- Lyapunov and gradient -----------------------------------------------

            x_t = x_t*2 - 1
            V = self.model_lyap.lyapunov_func(x_t)      # (1, 1)
            self.dV = (V.detach().clone().cpu().numpy() - self.last_V.detach().clone().cpu().numpy())*self.control_freq
            self.last_V = V 

            if(self.dV < 0):
                print("Lyap OK")
                #print(self.dV[0][0])
                self.Vdot_filtered = self.dV[0][0]
                return u_nominal
            
            else:
                print("Lyap not OK")
                return u_safe
            


            #print("Lyap not OK")

            

            V_scalar = V.squeeze()                             # scalar
            grad_V  = torch.autograd.grad(V_scalar, x_t, retain_graph=True,
                                        create_graph=True)[0].view(-1)  # (10,)

            grad_V_np = grad_V.detach().cpu().numpy()
            
            grad_V_x = np.concatenate([grad_V_np[:8], np.array([grad_V_np[9]])])
            grad_V_u = grad_V_np[8]         # scalar
            self.grad_u = grad_V_u

            # --------------------------------------------------- 4.  Early exits
            if abs(grad_V_u) < grad_thresh:
                print("🔹 |∂V/∂u| too small — skipping QP")
                return u_safe

            # --------------------------------------------------- 5.  Build RHS

            numpy_xt = x_t.detach().cpu().numpy()[0][0]
            numpy_xt = np.delete(numpy_xt,8)            
            
            Vdot_nom = np.dot(grad_V_x, (numpy_xt - self.last_xt)*(self.control_freq))

            self.last_xt = numpy_xt


            lyap_margin = max(delta_gain * abs(V.item()), delta_min)
            #lyap_margin = delta_min
            rhs = -lyap_margin          # want  V̇  ≤  −margin
            rhs = min(rhs, -delta_min)  # never LESS strict than −delta_min

            rhs = -10

            rhs = -1 * abs(V.item())

            #rhs = 0

            # --------------------------------------------------- 6. QP
            bias = -np.sign(grad_V_u) * 0  # small push in safe direction
            Δu = cp.Variable(u_flat.shape[0])
            objective = cp.Minimize(cp.sum_squares(Δu - bias))

            constr = [
                grad_V_u *(self.control_freq)* Δu + Vdot_nom <= rhs,
            ]
            obj = cp.Minimize(cp.sum_squares(Δu - bias))
            prob = cp.Problem(obj, constr)

            # —— solve
            try:
                prob.solve(solver=cp.OSQP, eps_abs=0.001, eps_rel=0.001,
                        verbose=False)
            except cp.error.SolverError:
                prob.solve(solver=cp.ECOS, eps_abs=0.001, eps_rel=0.001,
                        verbose=False)

            # --------------------------------------------------- 7.  Fallback
            if prob.status not in {"optimal", "optimal_inaccurate"}:
                print("⚠️ QP failed — using nominal")
                return u_safe
            
            self.print(Δu.value)

            value = np.clip(Δu.value/2,-du_clip,du_clip)

            # --------------------------------------------------- 8.  Return filtered action
            u_corr = (u_nominal.detach().cpu().numpy() + value).squeeze()
            #u_corr = (self.last_action + value).squeeze()

            self.last_action_filter = u_corr
            
            delta = Δu.value/2

            # ---------- Diagnostics ----------
            self.V = V.item()
            self.vdot = Vdot_nom
            if self.print_flag:
                self.print(f"V(x)               : {V.item():.6f}")
                self.print(f"∂V/∂u              : {grad_V_u:+.5f}")
                self.print(f"‖∂V/∂x‖            : {np.linalg.norm(grad_V_x):.5f}")
                self.print(f"V̇_nominal         : {Vdot_nom:+.6f}")
                self.print(f"Target RHS         : {rhs:.6f}")
                self.print(f"u_nominal / filtered: {u_nominal.item():.4f} → {u_corr:.4f}")
                self.print(delta)
            # ----------------------------------

            u_corr = torch.tensor(u_corr, dtype=torch.float32,
                            device=self.device,
                            requires_grad=True).squeeze(0) #[0,1]
            
            self.Vdot_filtered = grad_V_u*self.control_freq* np.clip(Δu.value,-du_clip,du_clip) + Vdot_nom
            self.print(self.Vdot_filtered)

            return u_corr

    def FL_controller(self, state_np_not_normalized, ref, last_ref, integral_error, last_action, 
                      Kp = 30, Ki = 10, timestep_duration = 0.001):
        F_actuator = state_np_not_normalized[0]
        F_load = state_np_not_normalized[1]
        Pa = state_np_not_normalized[2]  
        Pb = state_np_not_normalized[3]
        x_pos = state_np_not_normalized[4]
        x_deriv = state_np_not_normalized[5]
        F_load_deriv = state_np_not_normalized[6]
        F_actuator_deriv = state_np_not_normalized[7]

        #Model parameters
        vpl = [0.00121]
        Be = [1.34e9]
        pn = 7000000             
        qn = 10/60000             
        In = 0.050  
        L = 0.08 
        Kv = qn/(In*(pn/2)**(0.5))
        pt = 0
        ps = 16000000 
        alpha = 0.601
        Ap = 2.01e-4

        #Model calculation
        va = Ap*x_pos + vpl
        vb = alpha*Ap*(L - x_pos) + vpl

        vel_term = (ref - last_ref)/timestep_duration + Be*Ap*Ap*(alpha*alpha/vb + 1/va)*x_deriv

        g1 = Be*Ap*Kv*(((abs(ps - Pa))**(0.5))/va + alpha*((abs(Pb - pt))**(0.5))/vb)
        g2 = Be*Ap*Kv*(((abs(Pa- pt))**(0.5))/va + alpha*((abs(ps - Pb))**(0.5))/vb)

        if(last_action >= 0):
            g = g1
        else:
            g = g2
        
        integral_error = integral_error + (ref - F_load) * timestep_duration
        error_force = ref - F_load

        action_control = (vel_term + (error_force)*(Kp) + (integral_error)*(Ki))/(g)
        action_control = np.clip(self.action_control, -0.05, 0.05)  # Ensure action is within bounds
        action_control = action_control/0.05  # Adjust to match the action space range [-1, 1]
        action_control = action_control/2 + 0.5  

        action_control_normalized_zero_to_one = action_control

        return action_control_normalized_zero_to_one

    def step(self, action):
        applied_action = 0
        # === Apply Action Smoothing ===
        #action = float(np.clip(action, -1, 1))*self.action_scale

        #action = self.smooth_factor * self.last_action_rl + (1 - self.smooth_factor) * action

        self.action_filter = action

        #action = action/2 + 0.5 # Scale to [-1, 1] range

        #Controller action calculation
        keys = ['F_actuator', 'F_actuator_deriv', 'F_load', 'F_load_deriv',
                'Pa', 'Pb', 'x', 'x_deriv']  # last_x shares x scale
        
        self.mins = [self.get_norm(k)[0] for k in keys]
        self.maxs = [self.get_norm(k)[1] for k in keys]

        self.mins_lyap = [self.get_norm_lyap(k)[0] for k in keys]
        self.maxs_lyap = [self.get_norm_lyap(k)[1] for k in keys]

        mins = [self.get_norm(k)[0] for k in keys]
        maxs = [self.get_norm(k)[1] for k in keys]

        x_denorm = self.current_state * torch.tensor([maxs[i] - mins[i] for i in range(8)], device=self.current_state.device) + \
                   torch.tensor(mins, device=self.current_state.device)
        

        (F_actuator, F_actuator_deriv, F_load, F_load_deriv,
         Pa, Pb, x_pos, x_deriv) = [x_denorm[:, j] for j in range(8)]
        
        F_actuator = F_actuator.item()
        F_load = F_load.item()
        Pa = Pa.item()  
        Pb = Pb.item()
        x_pos = x_pos.item()
        x_deriv = x_deriv.item()
        F_load_deriv = F_load_deriv.item()
        F_actuator_deriv = F_actuator_deriv.item()

        F_load_norm = self.current_state_bb.squeeze()[2].item()
        min_force, max_force = self.get_norm('F_load')
        F_load = F_load_norm * (max_force - min_force) + min_force

        desired_force = self.Force_desired[self.current_time_step][0]
        min_force, max_force = self.get_norm('F_load')
        desired_force_norm = (desired_force - min_force) / (max_force - min_force)

        min_force, max_force = self.get_norm('F_load')
        F_load = F_load_norm * (max_force - min_force) + min_force

        min_error, max_error = self.get_norm('F_load')

        force_error = desired_force - F_load
        force_error_norm = ((desired_force - min_force)/(max_force - min_force)) - ((F_load - min_force)/(max_force - min_force))
        #print(f"Force Error: {force_error_norm:.4f}")


        #FL
        if(True):
            #vpl_rl = ((action[0] + 1)/2)*(self.Max_vpl - self.Min_vpl) + self.Min_vpl
            #Kv_rl = ((action[1] + 1)/2)*(self.Max_Kv - self.Min_Kv) + self.Min_Kv
            #Be_rl = ((action[2] + 1)/2)*(self.Max_Be - self.Min_Be) + self.Min_Be
            Kp_rl = ((action[0] + 1)/2)*(self.Max_Kp - self.Min_Kp) + self.Min_Kp
            Ki_rl = ((action[1] + 1)/2)*(self.Max_Ki - self.Min_Ki) + self.Min_Ki



            vpl_rl = self.vpl.item()
            Be_rl = self.Be.item()
            Kv_rl = self.Kv.item()
            #Kp_rl = 0
            #Ki_rl = 0

            self.va = self.Ap*x_pos + vpl_rl
            self.vb = self.alpha*self.Ap*(self.L - x_pos) + vpl_rl

            vel_term = (self.Force_desired[self.current_time_step] - self.Force_desired[self.current_time_step - 1])/self.timestep_duration + self.Be.item()*self.Ap*self.Ap*(self.alpha*self.alpha/self.vb + 1/self.va)*x_deriv

            g1 = Be_rl*self.Ap*Kv_rl*(((abs(self.ps - Pa))**(0.5))/self.va + self.alpha*((abs(Pb - self.pt))**(0.5))/self.vb)
            g2 = Be_rl*self.Ap*Kv_rl*(((abs(Pa- self.pt))**(0.5))/self.va + self.alpha*((abs(self.ps - Pb))**(0.5))/self.vb)

            if(self.last_action >= 0):
                g = g1
            else:
                g = g2
            
            self.integral_error = self.integral_error + (self.Force_desired[self.current_time_step] - F_load) * self.timestep_duration
            error_force = self.Force_desired[self.current_time_step] - F_load

            self.action_control = (vel_term + (error_force)*(15 + Kp_rl) + (self.integral_error)*(5 + Ki_rl))/(g)
            self.action_control = np.clip(self.action_control, -0.05, 0.05)  # Ensure action is within bounds
            self.action_control = self.action_control.item()/0.05  # Adjust to match the action space range [-1, 1]
            self.action_control = self.action_control/2 + 0.5  

            action_control_safe = (vel_term + (error_force)*(15) + (self.integral_error)*(5))/(g)
            action_control_safe = np.clip(action_control_safe, -0.05, 0.05)  # Ensure action is within bounds
            action_control_safe = action_control_safe.item()/0.05  # Adjust to match the action space range [-1, 1]
            action_control_safe = action_control_safe/2 + 0.5 


            self.action_control_rl = ((error_force)*(Kp_rl) + (self.integral_error)*(Ki_rl))/(g)
            self.action_control_rl = np.clip(self.action_control_rl, -0.05, 0.05)  # Ensure action is within bounds
            self.action_control_rl = self.action_control_rl.item()/0.05  # Adjust to match the action space range [-1, 1]
            self.action_control_rl = self.action_control_rl/2 + 0.5 

        action_full = self.action_control  # Adjust action to include control component

        action_full = np.clip(action_full, 0, 1)  # Ensure action is within bounds

        action_full = float(action_full)  # Convert to float for consistency

        self.print(f"Action Control: {self.action_control:.4f}, Action Full: {action_full:.4f}")

        ##########################################################################################


        self.state_dummy = torch.cat((
            self.current_state_bb.squeeze(),
            torch.tensor([self.last_action], dtype=torch.float32, device=self.device),
            torch.tensor([force_error_norm], dtype=torch.float32, device=self.device)
        )).unsqueeze(0)

        # Convert current state to numpy (as expected by qp_safe_control)
        x_np = self.state_dummy

        # Ensure action_full is a proper torch tensor on the correct device with grad disabled
        if isinstance(action_full, torch.Tensor):
            u_nominal = action_full.to(dtype=torch.float32, device=self.device).unsqueeze(0)
            u_safe = action_control_safe.to(dtype=torch.float32, device=self.device).unsqueeze(0)
        else:
            u_nominal = torch.tensor([[action_full]], dtype=torch.float32, device=self.device)
            u_safe  = torch.tensor([[action_control_safe]], dtype=torch.float32, device=self.device)
            

        if(self.safe_control_strategy == 'None'):
            action_full = u_nominal.item()


        elif(self.safe_control_strategy == 'QP'):
            # Call the safe control method
            action_full = self.qp_safe_control(
                x_np,
                u_nominal,
                u_safe
            )


            action_full = action_full.item()

        #action_full = action_full.item()  # Convert to float for consistency

        action_full = np.clip(action_full, 0, 1)  # Ensure action is within bounds

        action_full = self.smooth_factor * self.last_action + (1 - self.smooth_factor) * action_full

        self.action_filter_after_qp = action_full

        applied_action = action_full
        self.last_action = applied_action
        self.last_action_rl = action     
        

        self.current_state_bb = torch.cat((
            self.current_state_bb.squeeze(),
            torch.tensor([applied_action], dtype=torch.float32, device=self.device)
        )).unsqueeze(0)
        

        # === Run both models ===
        output_bb = self.model.single_step(self.current_state_bb)
    
        self.current_state_bb = output_bb.unsqueeze(0)

        # === Combine outputs ===
        combined_state = output_bb

        self.current_state = combined_state + random.uniform(-0.005, 0.005)

        # === Denormalize Force ===
        F_load_norm = self.current_state.squeeze()[2].item()
        min_force, max_force = self.get_norm('F_load')
        F_load = F_load_norm * (max_force - min_force) + min_force

        # === Compute Reward ===
        desired_force = self.Force_desired[self.current_time_step][0]
        desired_force_norm = (desired_force - min_force) / (max_force - min_force)
        F_load_norm = (F_load - min_force) / (max_force - min_force)
        force_error = desired_force_norm - F_load_norm
        
        reward_boost = 0

        if(abs(desired_force - F_load) < 10):
            reward_boost = 1

        reward = -(
            (force_error ** 2) * self.Q +
            (action.mean() ** 2) * self.R +
            ((force_error - self.last_error) ** 2) * self.R2 +
            ((action.mean() - self.last_action_rl.mean()) ** 2) * self.R2
        )

        reward = reward + reward_boost

        self.last_error = force_error

        # === Check Termination ===
        terminated = False
        truncated = self.calculate_end()

        if terminated:
            reward = -1e5
            #print(f"❌ Terminated at timestep {self.current_time_step} with force error {force_error:.2f}")

        self.current_time_step += 1

        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
