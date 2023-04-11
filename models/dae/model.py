from pdb import set_trace
import torch
from torch import nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint
import torch.nn.functional as F
from scipy.io import mmread
from torch.utils.data import TensorDataset, DataLoader
import os
import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", "Experiment logs directory .* exists and is not empty.*")

# TODO: Find a way to make hidden dimensions dynamic
class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim, h_dim=(200, 150, 100, 50)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, h_dim[0]),
            nn.ReLU(),
            nn.Linear(h_dim[0], h_dim[1]),
            nn.ReLU(),
            nn.Linear(h_dim[1], h_dim[2]),
            nn.ReLU(),
            nn.Linear(h_dim[2], h_dim[3]),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(h_dim[3], h_dim[2]),
            nn.ReLU(),
            nn.Linear(h_dim[2], h_dim[1]),
            nn.ReLU(),
            nn.Linear(h_dim[1], h_dim[0]),
            nn.ReLU(),
            nn.Linear(h_dim[0], input_dim)
        )
    
    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class LitDeepAutoEncoder(pl.LightningModule):
    def __init__(self, AutoEncoder) -> None:
        super().__init__()
        self.encoder = AutoEncoder.encoder
        self.decoder = AutoEncoder.decoder
        
    def forward(self, x) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x
        
    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_reconstructed = self(x)
        loss = F.mse_loss(x_reconstructed, x)
        self.log('train_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
    
if __name__ == "__main__":
    # Matrix of dimension (training data points, data point length)
    data = mmread("../../data/encoded_antibodies.mtx")
    data = torch.tensor(data, dtype=torch.float32)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # Settings for training
    # TODO: Make this dynamic so it recognizes CUDA, MPS, etc. and falls back to CPU
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "CPU")
    INPUT_DIM = data.shape[1]
    
    # specify the checkpoint directory and file name
    # TODO: Improve callback to better control model versioning
    checkpoint_callback = ModelCheckpoint(
        dirpath='logs/dae/version_0/checkpoints',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_last=True  # save the last checkpoint only
    )
    
    # Train the model
    autoencoder = LitDeepAutoEncoder(DeepAutoEncoder(INPUT_DIM))
    trainer = pl.Trainer(
        max_epochs=100, 
        log_every_n_steps=5, 
        logger=pl.loggers.CSVLogger("logs", name="dae"),
        callbacks=[checkpoint_callback],
    )
    trainer.fit(
        model=autoencoder, 
        train_dataloaders=dataloader
    )