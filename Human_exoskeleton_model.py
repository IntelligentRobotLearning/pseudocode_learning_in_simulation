""" Neural network architecture definition for exoskeleton, human and muscles"""

import torch.nn as nn
import numpy as np

MultiVariateNormal = torch.distributions.Normal

class MuscleNN(nn.Module):
    def __init__(self, num_total_muscle_related_dofs, num_dofs, num_muscles):
        super(MuscleNN, self).__init__()
        self.num_total_muscle_related_dofs = num_total_muscle_related_dofs
        self.num_dofs = num_dofs
        self.num_muscles = num_muscles

        self.fc = nn.Sequential(
            nn.Linear(num_total_muscle_related_dofs + num_dofs, num_h1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, num_h2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_h3),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, num_muscles),
            nn.Tanh(),
        )
        self.std_muscle_tau = torch.zeros(self.num_total_muscle_related_dofs)
        self.std_tau = torch.zeros(self.num_dofs)

        for i in range(self.num_total_muscle_related_dofs):
            self.std_muscle_tau[i] = 200.0

        for i in range(self.num_dofs):
            self.std_tau[i] = 200.0
        
    def forward(self, muscle_tau, tau):
        muscle_tau = muscle_tau / self.std_muscle_tau
        tau = tau / self.std_tau
        out = self.fc(torch.cat([muscle_tau, tau], dim=1))
        return nn.ReLU()(out)

    def get_activation(self, muscle_tau, tau):
        act = self.forward(muscle_tau.reshape(1, -1), tau.reshape(1, -1))
        return act.cpu().detach().numpy()


class SimulationExoNN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(SimulationExoNN, self).__init__()

        num_h1 = 128
        num_h2 = 64
        self.num_actions = num_actions
        self.policy = nn.Sequential(
            nn.Linear(num_states, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, num_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(num_states, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        p_out = self.policy(x)
        p_out = MultiVariateNormal(p_out, self.log_std.exp())
        v_out = self.value(x)
        return p_out, v_out

    def get_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()

class SimulationHumanNN(nn.Module):
    def __init__(self, num_states, num_actions):
        super(SimulationHumanNN, self).__init__()

        num_h1 = 256
        num_h2 = 256
        self.num_actions = num_actions
        self.policy = nn.Sequential(
            nn.Linear(num_states, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, num_actions)
        )
        self.value = nn.Sequential(
            nn.Linear(num_states, num_h1),
            nn.ReLU(),
            nn.Linear(num_h1, num_h2),
            nn.ReLU(),
            nn.Linear(num_h2, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(num_actions))

    def forward(self, x):
        p_out = self.policy(x)
        p_out = MultiVariateNormal(p_out, self.log_std.exp())
        v_out = self.value(x)
        return p_out, v_out

    def get_action(self, s):
        ts = torch.tensor(s)
        p, _ = self.forward(ts)
        return p.loc.cpu().detach().numpy()
