#
# Trains an MNIST Autoencoder using PyTorch Lightning,
# and uses MLflow to log metrics, params and artifacts
# NOTE: This example requires you to first install
# pytorch-lightning (using pip install pytorch-lightning)
#       and mlflow (using pip install mlflow).
#


import os

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.cli import LightningCLI
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from dotenv import load_dotenv

import mlflow.pytorch
from mlflow.tracking import MlflowClient
from mlflow.models import infer_signature


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=64, num_workers=3):
        """
        Initialization of inherited lightning data module
        """
        super().__init__()
        self.df_train = None
        self.df_val = None
        self.df_test = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.test_data_loader = None
        self.batch_size = batch_size
        self.num_workers = num_workers

        # transforms for images
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    def setup(self, stage=None):
        """
        Downloads the data, parse it and split the data into train, test, validation data

        Args:
            stage: Stage - training or testing
        """

        self.df_train = datasets.MNIST(
            "dataset", download=True, train=True, transform=self.transform
        )
        self.df_train, self.df_val = random_split(self.df_train, [55000, 5000])
        self.df_test = datasets.MNIST(
            "dataset", download=True, train=False, transform=self.transform
        )

    def create_data_loader(self, df):
        """
        Generic data loader function

        Args:
            df: Input tensor

        Returns:
            Returns the constructed dataloader
        """
        return DataLoader(df, batch_size=self.batch_size, num_workers=self.num_workers)

    def train_dataloader(self):
        """
        Returns:
            output: Train data loader for the given input.
        """
        return self.create_data_loader(self.df_train)

    def val_dataloader(self):
        """
        Returns:
            output: Validation data loader for the given input.
        """
        return self.create_data_loader(self.df_val)

    def test_dataloader(self):
        """
        Returns:
            output: Test data loader for the given input.
        """
        return self.create_data_loader(self.df_test)


class Autoencoder(L.LightningModule):
    def __init__(self, input_dim=28*28, latent_dim=32, learning_rate=0.01):
        """
        Initializes the network
        """
        super().__init__()
        self.optimizer = None
        self.scheduler = None
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
        self.learning_rate = learning_rate
        self.val_outputs = []
        self.test_outputs = []

    def forward(self, x):
        """
        Args:
            x: Input mnist image

        Returns:
            output - reconstructed mnist image
        """
        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(x.size(0), -1)
        latent_vector = self.encoder(x)
        reconstructed = self.decoder(latent_vector)
        reconstructed = reconstructed.view(-1, 28, 28)
        return reconstructed

    def mse_loss(self, reconstructed, raw):
        """
        Initializes the loss function

        Returns:
            output: Initialized mse loss function.
        """
        loss = nn.MSELoss()(reconstructed, raw)
        return loss

    def training_step(self, train_batch, batch_idx):
        """
        Training the data as batches and returns training loss on each batch

        Args:
            train_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output - Training loss
        """
        x, _ = train_batch
        x = x.view(x.size(0), -1)
        reconstructed = self.forward(x)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)
        loss = self.mse_loss(reconstructed, x)
        return {"loss": loss}

    def validation_step(self, val_batch, batch_idx):
        """
        Performs validation of data in batches

        Args:
            val_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output: valid step loss
        """
        x, _ = val_batch
        x = x.view(x.size(0), -1)
        reconstructed = self.forward(x)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)
        loss = self.mse_loss(reconstructed, x)
        self.val_outputs.append(loss)
        return {"val_step_loss": loss}

    def on_validation_epoch_end(self):
        """
        Computes average validation loss
        """
        avg_loss = torch.stack(self.val_outputs).mean()
        self.log("val_loss", avg_loss, sync_dist=True)
        self.val_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        """
        Performs test and computes the metric of the model

        Args:
            test_batch: Batch data
            batch_idx: Batch indices

        Returns:
            output: Testing metric
        """
        x, _ = test_batch
        x = x.view(x.size(0), -1)
        reconstructed = self.forward(x)
        reconstructed = reconstructed.view(reconstructed.size(0), -1)
        # Compute cosine similarity
        cos_sim = F.cosine_similarity(x, reconstructed, dim=0)
        self.test_outputs.append(cos_sim)
        return {"test_cos_sim": cos_sim}

    def on_test_epoch_end(self):
        """
        Computes average test accuracy score
        """
        avg_cos_sim = torch.stack(self.test_outputs).mean()
        self.log("avg_test_cos_sim", avg_cos_sim, sync_dist=True)
        self.test_outputs.clear()

    def configure_optimizers(self):
        """
        Initializes the optimizer and learning rate scheduler

        Returns:
            output: Initialized optimizer and scheduler
        """
        #self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        self.scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=0.2,
                patience=2,
                min_lr=1e-6,
                verbose=True,
            ),
            "monitor": "val_loss",
        }
        return [self.optimizer], [self.scheduler]

def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print(f"run_id: {r.info.run_id}")
    print(f"artifacts: {artifacts}")
    print(f"params: {r.data.params}")
    print(f"metrics: {r.data.metrics}")
    print(f"tags: {tags}")

def cli_main():
    early_stopping = EarlyStopping(
        monitor="val_loss",
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.getcwd(), save_top_k=1, verbose=True, monitor="val_loss", mode="min"
    )
    lr_logger = LearningRateMonitor()
    cli = LightningCLI(
        Autoencoder,
        MNISTDataModule,
        run=False,
        save_config_callback=None,
        trainer_defaults={
            "callbacks": [early_stopping, checkpoint_callback, lr_logger],
            "max_epochs": 3,
            },
    )

    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    cli.trainer.test(ckpt_path="best", datamodule=cli.datamodule)
   
    # Get a sample input from the test dataset
    sample_input, _ = next(iter(cli.datamodule.test_dataloader()))
    
    # Pass the sample input through the model to get the output
    with torch.no_grad():
        sample_output = cli.model(sample_input).detach()

    return cli, sample_input, sample_output
    

if __name__ == "__main__":
    # Load environment variables from .env file
    load_dotenv()

    # Access the environment variables
    mlflow_port = os.getenv("MLFLOW_PORT")

    mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")
    # Create a new MLflow Experiment
    mlflow.set_experiment("Pytorch Autoencoder Demo")
    # Start an MLflow run
    with mlflow.start_run(run_name = 'test1') as run:
        # Auto log all MLflow entities
        mlflow.pytorch.autolog(
            log_every_n_epoch=1,
            log_models=False, 
            log_datasets=True,
            extra_tags={
                "Flavor": "Pytorch",
                "Log Mode": "Manual",
                },
            )
        
        cli, sample_input, sample_output = cli_main()
        # Convert both input and output to NumPy arrays, as mlflow Tensor-based schemas support numpy data types only.
        sample_input = sample_input.numpy()
        sample_output = sample_output.numpy()
        # Infer the signature
        signature = infer_signature(sample_input, sample_output)
        mlflow.pytorch.log_model(cli.model, "MNIST Autoencoder", registered_model_name = 'MNIST Autoencoder', signature=signature)
    # Fetch the auto logged parameters and metrics.
    print_auto_logged_info(mlflow.get_run(run_id=run.info.run_id))

