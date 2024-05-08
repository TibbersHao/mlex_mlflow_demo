import mlflow.pytorch
import torch

mlflow.set_tracking_uri("http://localhost:5000")
# Load your PyTorch models
model1 = torch.load("models/uid0013_MSDNet.pt")
model2 = torch.load("models/uid0014_MSDNet.pt")
model3 = torch.load("models/uid0015_TUNet.pt")

print(type(model1.items()))
# Start an MLflow run
with mlflow.start_run():

    # Log the first PyTorch model with versioning and an alias
    mlflow.pytorch.log_model(model1, "msdnet", registered_model_name="msdnet_maxdil")
    mlflow.pytorch.log_model(model1, "msdnet", registered_model_name="msdnet_customdil")
    mlflow.pytorch.log_model(model2, "tunet", registered_model_name="tunet)

    # Create aliases for the models
    mlflow.create_model_version("msdnet", "1", "max_dilation")
    mlflow.create_model_version("msdnet", "1", "custom_dilation")
    mlflow.create_model_version("tunet", "1", "development")