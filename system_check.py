import torch, time

# checking system
print("PyTorch:", torch.__version__)
print("MPS built:", torch.backends.mps.is_built())
print("MPS available:", torch.backends.mps.is_available())
device = (
    torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cuda" if torch.cuda.is_available() else "cpu")
)
print("Using device:", device)

x = torch.randn(5000, 5000, device="mps")
torch.matmul(x, x)
torch.mps.synchronize()
t0 = time.time()
for _ in range(10):
    torch.matmul(x, x)
torch.mps.synchronize()
print("Time for 10 matrix mults:", time.time()-t0, "s")

x = torch.randn(5000, 5000, device="cpu")
torch.matmul(x, x)
torch.mps.synchronize()
t0 = time.time()
for _ in range(10):
    torch.matmul(x, x)
torch.mps.synchronize()
print("Time for 10 matrix mults:", time.time()-t0, "s")