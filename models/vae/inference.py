from os import path
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import pdb
import torch
from scipy.io import mmread
import pathlib
from argparse import ArgumentParser

# Argument Parsing
parser = ArgumentParser(
    description = "Variational AutoEncoder Inference script",
    epilog = "See https://github.com/dgorhe/antibody-design for usage and instructions"
)

# TODO: Add random permutation flag for permuting subset selection
parser.add_argument("-d", "--data", default="../../data/encoded_antibodies.mtx", help='File path for data to run inference on. Should be mtx file of dimension (inference examples, example size)')
parser.add_argument("-m", "--model", type=pathlib.Path, default="trained/train_model_test.pt", help='File path for trained model.')
parser.add_argument("-s", "--subset", type=int, default=10, help='Number of rows to run inference on (row 1 through s)')
parser.add_argument("-n", "--name", type=pathlib.Path, default="results/results.csv", help='Name of output csv file')
args = parser.parse_args()

print("Loading model")
model = torch.load(args.model)
model.eval()

print("Loading inference data")
inf_data = torch.tensor(mmread(args.data), dtype=torch.float32)
subset = args.subset if args.subset is not None else inf_data.shape[0]

df = pd.DataFrame(columns=['x', 'x_hat'])
decoder = json.load(open("../../data/decoding.json", "r"))
int2symbol = decoder["symbol"]["one-character"]

for i in tqdm(range(subset), total=subset):
    # Perform inference
    x = inf_data[i]
    x_hat, mu, sigma = model(x)
    
    # Convert tensors into numpy arrays
    x_seq = x.detach().numpy().astype(np.int8).tolist()
    x_seq = "".join([int2symbol[str(i)] for i in x_seq])
    x_hat_seq = x_hat.detach().numpy().astype(np.int8).tolist()
    x_hat_seq = "".join([int2symbol[str(i)] for i in x_hat_seq])
    
    mu = mu.detach().numpy(force=True)
    sigma = sigma.detach().numpy(force=True)
    
    # Save results to dataframe
    row = pd.Series({"x": x_seq, "x_hat": x_hat_seq})
    df = pd.concat([df, row.to_frame().T], ignore_index=True)

print(f"Saving results to: \n{path.abspath(args.name)}")
df.to_csv(args.name)


