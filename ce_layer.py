import torch
import torch.nn as nn

class CELayer(nn.Module):
    """
    CE Layer: Implements Consciousness Equation CE = I × A × Θ
    where:
      - I = Information tensor
      - A = Attention weighting
      - Θ = Integration factor

    This minimal implementation projects the input to three parallel streams,
    applies elementwise nonlinearities, then multiplies them together to produce
    a CE-modulated output tensor with the same shape as input.
    """

    def __init__(self, dim):
        super(CELayer, self).__init__()
        # Linear transforms for each component
        self.info_proj = nn.Linear(dim, dim)
        self.attn_proj = nn.Linear(dim, dim)
        self.theta_proj = nn.Linear(dim, dim)

        # Optional gating parameter (learnable scalar) to scale CE contribution
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        """
        x: [batch, seq, dim] input tensor
        returns: [batch, seq, dim] CE-modulated tensor
        """
        # Project
        I = torch.sigmoid(self.info_proj(x))     # information in [0,1]
        A = torch.sigmoid(self.attn_proj(x))     # attention in [0,1]
        Theta = torch.tanh(self.theta_proj(x))   # integration in [-1,1]

        CE = I * A * Theta                       # elementwise CE
        return self.alpha * CE
