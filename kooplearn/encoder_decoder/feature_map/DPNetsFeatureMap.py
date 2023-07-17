from kooplearn.encoder_decoder.feature_map.FeatureMap import FeatureMap
import lightning as L
import torch
from kooplearn.encoder_decoder.DPNetsModel import EncoderDecoderModel
from kooplearn.data.utils.TimeseriesDataModule import TimeseriesDataModule
from kooplearn.encoder_decoder.nn.modules.DPNetsModule import DPNetsModule


class DPNetsFeatureMap:
    def __init__(self,
                 dnn_model_class, dnn_model_kwargs,
                 optimizer_fn, optimizer_kwargs,
                 scheduler_fn, scheduler_kwargs, scheduler_config,
                 callbacks_fns, callbacks_kwargs,
                 logger_fn, logger_kwargs,
                 trainer_kwargs,
                 seed,
                 loss_fn,
                 dnn_model_class_2=None, dnn_model_kwargs_2=None,
                 ):
        self.dnn_model_module_class = DPNetsModule
        self.dnn_model_class = dnn_model_class
        self.dnn_model_kwargs = dnn_model_kwargs
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.scheduler_fn = scheduler_fn
        self.scheduler_kwargs = scheduler_kwargs
        self.scheduler_config = scheduler_config
        self.callbacks_fns = callbacks_fns
        self.callbacks_kwargs = callbacks_kwargs
        self.logger_fn = logger_fn
        self.logger_kwargs = logger_kwargs
        self.trainer_kwargs = trainer_kwargs
        self.loss_fn = loss_fn
        self.seed = seed
        self.dnn_model_class_2 = dnn_model_class_2
        self.dnn_model_kwargs_2 = dnn_model_kwargs_2
        L.seed_everything(seed)
        self.logger = None
        self.datamodule = None
        self.dnn_model_module = None
        self.callbacks = None
        self.trainer = None
        self.model = None

    def initialize_logger(self):
        self.logger = self.logger_fn(**self.logger_kwargs)
        # log what is not logged by default using pytorch lightning
        self.logger.log_hyperparams({'seed': self.seed})
        self.logger.log_hyperparams(self.trainer_kwargs)
        for kwargs in self.callbacks_kwargs:
            self.logger.log_hyperparams(kwargs)

    def initialize_model_module(self):
        self.dnn_model_module = self.dnn_model_module_class(
            model_class=self.dnn_model_class, model_hyperparameters=self.dnn_model_kwargs,
            optimizer_fn=self.optimizer_fn, optimizer_hyperparameters=self.optimizer_kwargs, loss_fn=self.loss_fn,
            scheduler_fn=self.scheduler_fn, scheduler_hyperparameters=self.scheduler_kwargs,
            scheduler_config=self.scheduler_config,
            model_class_2=self.dnn_model_class_2, model_hyperparameters_2=self.dnn_model_kwargs_2,
        )

    def initialize_callbacks(self):
        self.callbacks = [fn(**kwargs) for fn, kwargs in zip(self.callbacks_fns, self.callbacks_kwargs)]

    def initialize_trainer(self):
        self.trainer = L.Trainer(**self.trainer_kwargs, callbacks=self.callbacks, logger=self.logger)

    def initialize(self):
        self.initialize_logger()
        self.initialize_model_module()
        self.initialize_callbacks()
        self.initialize_trainer()

    def fit(self, X=None, Y=None, datamodule_kwargs=None):
        if datamodule_kwargs is None:
            raise ValueError('Datamodule is required to use DNNFeatureMap.')
        self.datamodule = TimeseriesDataModule(**datamodule_kwargs)
        self.trainer.fit(self.dnn_model_module, self.datamodule)

    def __call__(self, X):
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.float32)
        self.dnn_model_module.eval()
        with torch.no_grad():
            model_output = self.dnn_model_module(X)
        return model_output['x_encoded'].detach().numpy()  # Everything should be outputted as a Numpy array
    # In the case where X is the entire dataset, we should implement a dataloader to avoid memory issues
    # (prediction on batches). For this we should implement a predict_step and call predict on the trainer.
