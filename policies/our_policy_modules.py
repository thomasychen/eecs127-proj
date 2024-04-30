import typing

import torch.nn as nn
from torch import Tensor
import torch
import torch.nn.functional as F

class M2P1LinearModule(nn.Module):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t.
    """
    def forward(self, y_history: typing.List[Tensor]):
        t = len(y_history)  # List[()] -> ()
        y_t: Tensor = y_history[-1]  # ()
        if t ==1:
            u_t: Tensor = self.theta + self.theta1 * y_t  # ()
        else:
            y_t_old = y_history[-2]
            u_t: Tensor = self.theta + self.theta1 * y_t + self.theta2*y_t_old
        return u_t


class FixedWeightM2P1LinearModule(M2P1LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is fixed at initialization.
    """
    def __init__(self, theta: float, theta1: float, theta2: float):
        super(FixedWeightM2P1LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.tensor(theta), requires_grad=False)
        self.theta1: nn.Parameter = nn.Parameter(data=torch.tensor(theta1), requires_grad=False)
        self.theta2: nn.Parameter = nn.Parameter(data=torch.tensor(theta2), requires_grad=False)


class LearnableWeightM2P1LinearModule(M2P1LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is trainable.
    Initializes theta_0 = 0.
    """
    def __init__(self):
        super(LearnableWeightM2P1LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)
        self.theta1: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)
        self.theta2: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)


class M1P2LinearModule(nn.Module):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t.
    """
    def forward(self, y_history: typing.List[Tensor]):
        t = len(y_history) - 1  # List[()] -> ()
        y_t: Tensor = y_history[-1]  # ()
        if t % 2 == 0:
            u_t: Tensor = self.theta + self.theta1 * y_t  # ()
        else:
            u_t: Tensor = self.theta2 + self.theta3 * y_t
        return u_t


class FixedWeightM1P2LinearModule(M1P2LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is fixed at initialization.
    """
    def __init__(self, theta: float, theta1: float, theta2: float, theta3:float):
        super(FixedWeightM1P2LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.tensor(theta), requires_grad=False)
        self.theta1: nn.Parameter = nn.Parameter(data=torch.tensor(theta1), requires_grad=False)
        self.theta2: nn.Parameter = nn.Parameter(data=torch.tensor(theta2), requires_grad=False)
        self.theta3: nn.Parameter = nn.Parameter(data=torch.tensor(theta3), requires_grad=False)


class LearnableWeightM1P2LinearModule(M1P2LinearModule):
    """
    Implements the policy function F(Y_(t)) = theta * Y_t, for theta a parameter whose value is trainable.
    Initializes theta_0 = 0.
    """
    def __init__(self):
        super(LearnableWeightM1P2LinearModule, self).__init__()
        self.theta: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)
        self.theta1: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)
        self.theta2: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)
        self.theta3: nn.Parameter = nn.Parameter(data=torch.zeros(size=()), requires_grad=True)




class SlidingWindowMLP(nn.Module):
    """
    MLP that processes a sliding window of the last 10 observations to produce a control signal u_t.
    """
    def __init__(self, input_size=5, hidden_size=10, output_size=1):
        super(SlidingWindowMLP, self).__init__()
        self.window = input_size
        
        # Define the first layer (input to hidden)
        self.hidden = nn.Linear(input_size, hidden_size)
        
        # Activation function for the hidden layer
        self.activation = nn.ReLU()
        
        # Define the second layer (hidden to output)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, y_history: typing.List[Tensor]):


        history: Tensor = y_history[-self.window:]
        padding_needed = self.window - len(history)
        padding = [0] * padding_needed
    
        history = padding + history
    
        history = torch.tensor(history, dtype=torch.float32)


        x = self.hidden(history)
        x = self.activation(x)
        u_t = self.output(x).squeeze()

        return u_t



__all__ = ["FixedWeightM2P1LinearModule", "LearnableWeightM2P1LinearModule", "FixedWeightM1P2LinearModule", "LearnableWeightM1P2LinearModule", "SlidingWindowMLP"]