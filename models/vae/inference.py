import pdb
import torch
from scipy.io import mmread

# TODO: Add argparse here

model = torch.load("trained/train_model_test.pt")
model.eval()

inf_data = mmread("../../data/encoded_antibodies.mtx")
test_subset = inf_data[0:5]

for i in range(5):
    x = torch.tensor(test_subset[i], dtype=torch.float32)
    x_hat, mu, sigma = model(x)
    print(f"Sample {i}:", x_hat.shape, mu.shape, sigma.shape)

