# Train by GPU and test by CPU
import torch


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

device = get_device()

# Assume this is GPU 
net = model.Net(XXX).to(device)

state = {
        "epoch": epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

torch.save(state, "XXXX.pt")

## test with CPU
checkpoint = torch.load("XXXX.pt",map_location=lambda storage, loc: storage)
net.load_state_dict(checkpoint["model_state_dict"])

output = net(input)
