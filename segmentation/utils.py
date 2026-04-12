import yaml
import argparse

import lightning.pytorch as pl
from torch.optim import Optimizer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.loggers import MLFlowLogger
# from aimstack.pytorch_lightning_tracker.loggers import BaseLogger as AimLogger



def merge_config(
    config_file: str, args: argparse.Namespace, model_config_file: str = None
) -> argparse.Namespace:
    """Merges YAML and argparse commands

    Args:
        config_file (str): Path to main YAML config
        model_config_file (str): Path to model specific YAML config
        args (argparse.Namespace): Command-line namespace

    Returns:
        argparse.Namespace: Merged configuration
    """
    with open(config_file, "r") as file:
        yml_config = yaml.safe_load(file)

    if model_config_file is not None:
        with open(model_config_file, "r") as file:
            yml_model_config = yaml.safe_load(file)

    # Remove None values
    args_dict = vars(args)
    args_dict = {k: v for k, v in args_dict.items() if v != None}

    # Merge the dictionaries, giving priority to YAML values
    # TODO: Probably gives priority to args, as it's unpacked last
    if model_config_file is not None:
        merged_config = {**yml_config, **yml_model_config, **args_dict}
    else:
        merged_config = {**yml_config, **args_dict}
    return argparse.Namespace(**merged_config)


def merge_model_params_segementation_encoder(
        config: argparse.Namespace,
        encoder_config_path
):
    """Merges the encoder parameters with the segmentation model parameters.

    Args:
        config (argparse.Namespace): Configuration namespace
        encoder_config_path (str): Path to the encoder configuration file

    Returns:
        argparse.Namespace: Merged configuration
    """
    with open(encoder_config_path, "r") as file:
        yml_encoder_config = yaml.safe_load(file)

    # Remove None values
    encoder_config_dict = yml_encoder_config["model_params"]
    encoder_config_dict = {k: v for k, v in encoder_config_dict.items() if v != None}

    # Update parameters that need to be consisten between pretrained encoder
    # and segmentation model.
    model_params = config.model_params
    model_params = {**encoder_config_dict, **config.model_params}
    model_params["vit_type"] = encoder_config_dict.get("vit_type", "b")
    model_params["reg_tokens"] = encoder_config_dict.get("reg_tokens", 0)
    model_params["patch_size"] = encoder_config_dict.get("patch_size", 14)
    model_params["in_chans"] = encoder_config_dict.get("in_chans", 2)
    model_params["double_view"] = encoder_config_dict.get("double_view", False)
    print("double view is ", model_params["double_view"])
    return model_params
    
    

def get_loggers(
    tracking_uri: str,
    job_id: str,
    logging_path: str,
    run_name: str,
    experiment_n: int,
    experiment_name: str = "aoi-vit",
    use_tb_logger: bool = False,
    use_csv_logger: bool = True, # Backup logger for MLFlow
) -> list[Logger]:
    """Returns a list of logger objects for experiment tracking, currently only supports
    MLFlow, Aim and Tensorboard.

    Args:
        tracking_uri (str): MLFlow URI for tracking
        job_id (str): Slurm identifier
        logging_path (str): Path to store local experiment files.
        experiment_n (int): Number of the experiment to add as tag to mlflow.
        experiment_name (str, optional): Name of the experiment. Defaults to "aoi-vit".
        use_tb_logger (bool, optional): Use tb logger for debugging.

    Returns:
        list[Logger]: List of PyTorch Lightning loggers
    """

    loggers = []

    mlflow_logger = SafeMLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name,
        tags={"slurm_id": job_id, "experiment_number": str(experiment_n)},
    )  # Sadly tags force values to be strings.

    loggers.append(mlflow_logger)
    if use_tb_logger:
        tb_logger = pl_loggers.TensorBoardLogger(
            str(logging_path), name=experiment_name
        )
        loggers.append(tb_logger)

    if use_csv_logger:
        csv_logger = pl_loggers.CSVLogger(
            str(logging_path), name="backup_csv_logs"
        )
        loggers.append(csv_logger)

    # else:
    #     aim_logger = AimLogger(
    #         experiment_name=experiment_name,
    #         run_name=mlflow_logger.run_id,
    #     )
    return loggers


def grad_norm(pl_module: pl.LightningModule) -> float:
    """Computes the L2 norm of the gradients of a model."""
    total_norm = 0.0
    for p in pl_module.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** (1.0 / 2.0)


class GradNormCallback(Callback):
    """Log all gradient norms before scaling, and clipping."""

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: Optimizer,
    ) -> None:
        pl_module.log("train/grad_norm", grad_norm(pl_module))



class SafeMLFlowLogger(MLFlowLogger):
    """ A wrapper around MLFlowLogger to handle exceptions gracefully.

    This class overrides the log_hyperparams and log_metrics methods to catch
    exceptions that may occur during logging. If an exception occurs, it prints
    an error message and continues execution without raising the exception so training
    is not completely halted.
    """
    def log_hyperparams(self, params):
        try:
            super().log_hyperparams(params)
        except Exception as e:
            print(f"Failed to log hyperparameters to MLflow: {e}. Ignoring this error.")

    def log_metrics(self, metrics, step=None):
        try:
            super().log_metrics(metrics, step)
        except Exception as e:
            print(f"Failed to log metrics to MLflow: {e}. Ignoring this error.")
