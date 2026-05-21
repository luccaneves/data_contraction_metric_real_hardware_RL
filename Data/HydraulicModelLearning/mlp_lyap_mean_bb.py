import torch
import torch.nn as nn
import torch.nn.functional as F

class Square(nn.Module):
    def forward(self, x):
        return x ** 2

class BB_Lyap_Mean(nn.Module):
    def __init__(self, hidden_size,ts, json_file_path=None, predict_horizon=100, n_samples=100):
        super(BB_Lyap_Mean, self).__init__()

        self.predict_horizon = predict_horizon
        self.n_samples = n_samples
        self.hidden_size = hidden_size
        self.ts = ts

        # =================
        # 🔗 LSTM Network
        # =================
        self.net_1 = nn.Sequential(
            nn.Linear(9, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 8)  # 8 means + 8 stds
        )


        # =================
        # 🔒 Lyapunov Network
        # =================


    
    def net(self, x):

        return self.net_1(x[:,0:9])

    # ================================
    # 🔧 Lyapunov Losses
    # ================================


    def get_means_stds(self, x):
        #torch.set_grad_enabled(True)
        output = self.net(x[:, :9])  # (batch_size, 16)

        # Split means and stds
        means = output[:, 0::2]                       # (batch_size, 8)
        stds = output[:, 1::2].clamp(min=1e-6)         # (batch_size, 8)

        return means, stds
    
    def single_step(self, x):
        output = self.net(x[:, :9])  # (batch_size, 16

        return output

    # ================================
    # 🚀 Forward Pass (Rollout)
    # ================================
    def forward(self, x):
        outputs = []


        initial_state = x[:, :10]
        i_sequence = x[:, 10:10 + self.predict_horizon]

        current_state = initial_state.clone()

        batch_size = x.size(0)

        for step in range(self.predict_horizon):
            output = self.single_step(2*current_state[:, :9] - 1)
            output = output/2 + 0.5 # Rescale to [0, 1]
            output = current_state[:, :8] + output*self.ts  # Euler integration with dt=0.001s


            outputs.append(output)
            

            next_i = i_sequence[:, step].unsqueeze(1)

            next_input = torch.column_stack((
                output[:, 0],    # F_actuator
                output[:, 1],    # F_actuator_deriv
                output[:, 2],    # F_load
                output[:, 3],    # F_load_deriv
                output[:, 4],    # Pa
                output[:, 5],    # Pb
                output[:, 6],    # x
                output[:, 7],    # x_deriv
                next_i.squeeze()    # next i
            ))

            current_state = next_input

        outputs = torch.cat(outputs, dim=1)  # (batch_size, predict_horizon * 8)

        return outputs