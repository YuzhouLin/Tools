# Save a torch sensor

output = net(input)

torch.save(output, 'file.pt')
torch.load('file.pt')
