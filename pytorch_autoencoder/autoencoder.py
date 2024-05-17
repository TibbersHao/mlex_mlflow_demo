import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import mlflow.pytorch

class Autoencoder(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Use sigmoid to ensure output is between 0 and 1 (like MNIST)
        )

    def forward(self, x):
        latent_code = self.encoder(x)
        reconstructed = self.decoder(latent_code)
        return reconstructed

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.view(x.size(0), -1)
        reconstructed = self.forward(x)
        loss = nn.MSELoss()(reconstructed, x)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=64):
        super().__init__()
        self.batch_size = batch_size
        self.train_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        torchvision.datasets.MNIST(root='./data', train=True, download=True)
        torchvision.datasets.MNIST(root='./data', train=False, download=True)

    def setup(self, stage=None):
        self.train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=torchvision.transforms.ToTensor(), download=False)
        self.test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=torchvision.transforms.ToTensor(), download=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

if __name__ == "__main__":
    mlflow.autolog()

    with mlflow.start_run():
        data_module = MNISTDataModule()
        autoencoder = Autoencoder(input_dim=28*28, latent_dim=64)

        trainer = pl.Trainer(max_epochs=10)
        trainer.fit(autoencoder, data_module)

        # After training, let's do inference on a few images
        test_loader = data_module.test_dataloader()
        images, labels = next(iter(test_loader))
        images = images.view(images.size(0), -1)
        reconstructions = autoencoder(images)
