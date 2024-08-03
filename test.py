import torch

k = 3
nut_a_real = torch.randn(k,requires_grad=True)
nut_a_imag = torch.randn(k, requires_grad=True)

bridge_a_real = torch.randn(k,requires_grad=True)
bridge_a_imag = torch.randn(k, requires_grad=True)

dispersion_a_real = torch.randn(k,requires_grad=True)
dispersion_a_imag = torch.randn(k, requires_grad=True)

nut_b_real = torch.randn(k,requires_grad=True)
nut_b_imag = torch.randn(k, requires_grad=True)

bridge_b_real = torch.randn(k,requires_grad=True)
bridge_b_imag = torch.randn(k, requires_grad=True)

dispersion_b_real = torch.randn(k,requires_grad=True)
dispersion_b_imag = torch.randn(k, requires_grad=True)

