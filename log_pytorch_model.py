import torch
import mlflow.pytorch

mlflow.set_tracking_uri("http://localhost:5000")

model = torch.nn.Linear(1, 1)

# Log the model
with mlflow.start_run() as run:
    mlflow.pytorch.log_model(model, "simpleNN", registered_model_name="simpleNN")
