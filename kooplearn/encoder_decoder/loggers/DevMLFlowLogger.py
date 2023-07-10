from pytorch_lightning.loggers import MLFlowLogger
from mlflow.entities import Param
from lightning_fabric.utilities.logger import _convert_params, _flatten_dict
from typing import Any, Dict, Union
from argparse import Namespace


class DevMLFlowLogger(MLFlowLogger):
    def __init__(self, *args, prefix_params='', disable_log_hyperparams=False, **kwargs):
        self.prefix_params = prefix_params
        self.disable_log_hyperparams = disable_log_hyperparams
        super().__init__(*args, **kwargs)

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        if self.disable_log_hyperparams:
            return
        params = _convert_params(params)
        params = _flatten_dict(params)

        # Truncate parameter values to 250 characters.
        # TODO: MLflow 1.28 allows up to 500 characters: https://github.com/mlflow/mlflow/releases/tag/v1.28.0
        params_list = [Param(key=(self.prefix_params + k), value=str(v)[:250]) for k, v in params.items()]

        # Log in chunks of 100 parameters (the maximum allowed by MLflow).
        for idx in range(0, len(params_list), 100):
            self.experiment.log_batch(run_id=self.run_id, params=params_list[idx: idx + 100])
