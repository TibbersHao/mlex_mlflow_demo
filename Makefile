include .env
export

# Initialize Tracking Server
init_server:
	mlflow server --host 127.0.0.1 --port $(MLFLOW_PORT)