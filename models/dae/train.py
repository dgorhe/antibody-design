import pdb
from datetime import date
from tqdm import tqdm
from scipy.io import mmread
import torch
from torch import optim, nn
from model import DeepAutoEncoder
from argparse import ArgumentParser

# Argument Parsing
parser = ArgumentParser(
    description = "Deep AutoEncoder training script",
    epilog = "See https://github.com/dgorhe/antibody-design for usage and instructions"
)

parser.add_argument("-d", "--data", default="../../data/encoded_antibodies.mtx", help='File path for training data. Should be mtx file of dimension (training examples, example size)')
parser.add_argument("-n", type=int, default=20, help='Number of epochs to train for')
parser.add_argument("-batchsize", type=int, default=64, help='Number of training data points per batch')
parser.add_argument("-name", type=str, default=None, help="Name of the model to save in trained/. Defaults todays date in the form YEAR-MONTH-DAY")
args = parser.parse_args()

# Matrix of dimension (training data points, data point length)
data = mmread(args.data)
data = torch.tensor(data, dtype=torch.float32)

# Settings for training
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = data.shape[1]
NUM_EPOCHS = args.n
BATCH_SIZE = args.batchsize
LR_RATE = 1e-3

# Instantiate model
model = DeepAutoEncoder(INPUT_DIM).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LR_RATE)
loss_fn = nn.MSELoss(reduction='sum')

for epoch in range(NUM_EPOCHS):
    for x in tqdm(data, total=data.shape[0], desc=f'Epoch {epoch}'):
        # Forward pass
        x.to(DEVICE)
        x_reconstructed = model(x)
        
        reconstruction_loss = loss_fn(x_reconstructed, x)
        
        # Backward pass
        loss = reconstruction_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
model_name = f"trained/{args.name}.pt" if args.name is not None else f"trained/{str(date.today())}.pt"
torch.save(model, model_name)