"""
Simple demo script that runs a forward pass through CELayer.
"""
import torch
from src.ce_layer import CELayer

def main():
    batch = 2
    seq = 8
    dim = 32

    x = torch.randn(batch, seq, dim)
    layer = CELayer(dim)
    out = layer(x)

    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("CE output mean:", out.mean().item())

if __name__ == "__main__":
    main()
