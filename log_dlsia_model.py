import mlflow.pyfunc
import torch
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variables
mlflow_port = os.getenv("MLFLOW_PORT")

mlflow.set_tracking_uri(f"http://127.0.0.1:{mlflow_port}")

# Sample parameters for testing
# params = {
#     "num_layers": 5,
#     "max_dilation": 10,
#     "num_epochs": 3,
# }

# Load your PyTorch models
model = torch.load("models/pytest2/pytest2_MSDNet1.pt")

print(type(model.items()))

# Start an MLflow run
with mlflow.start_run():

    model_info = mlflow.pyfunc.log_model(
        artifact_path = 'models/pytest2/',
        python_model = model
    )

    # Create aliases for the models
    # mlflow.create_model_version("msdnet", "1", "max_dilation")
