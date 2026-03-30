from model import CNN

model = CNN()
model.load_state_dict(torch.load("model.pth"))
model.eval()