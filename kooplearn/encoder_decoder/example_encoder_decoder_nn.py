from kooplearn.encoder_decoder.EncoderDecoderModel import EncoderDecoderModel
from kooplearn.encoder_decoder.feature_map.DNNFeatureMap import DNNFeatureMap
from kooplearn.encoder_decoder.koopman_estimators.ExtendedDMD import ExtendedDMD
from kooplearn.encoder_decoder.nn.modules.DPNetsModule import DPNetsModule
from kooplearn.encoder_decoder.nn.architectures.MLPModel import MLPModel
from kooplearn.encoder_decoder.nn.loss_fns.dpnets_loss import dpnets_loss
from kooplearn.encoder_decoder.loggers.DevMLFlowLogger import DevMLFlowLogger
from functools import partial
import numpy as np

num_features = 10
num_samples = 100
data = np.random.rand(num_samples, num_features)
X = data[:-1]
Y = data[1:]
loss_fn = partial(dpnets_loss, rank=5,  p_loss_coef=0, s_loss_coef=1.0, reg_1_coef=1.0, reg_2_coef=0)
feature_map = DNNFeatureMap(dnn_model_module_class=DPNetsModule,
                            dnn_model_class=MLPModel,
                            dnn_model_kwargs={'input_dim': num_features},
                            optimizer_fn=MLPModel.get_default_optimizer_fn(),
                            optimizer_kwargs=MLPModel.get_default_optimizer_kwargs(),
                            scheduler_fn=MLPModel.get_default_scheduler_fn(),
                            scheduler_kwargs=MLPModel.get_default_scheduler_kwargs(),
                            scheduler_config=MLPModel.get_default_scheduler_config(),
                            callbacks_fns=MLPModel.get_default_callbacks_fns(),
                            callbacks_kwargs=MLPModel.get_default_callbacks_kwargs(),
                            logger_fn=DevMLFlowLogger,
                            logger_kwargs=dict(prefix='pl', experiment_name='experiment_name', run_name='run_name'),
                            trainer_kwargs=MLPModel.get_default_trainer_kwargs(),
                            loss_fn=loss_fn,
                            seed=0,
                            )
decoder = feature_map.decoder_from_feature_map(num_features=num_features)
koopman_estimator = ExtendedDMD(feature_map=feature_map, rank=5)
model = EncoderDecoderModel(feature_map=feature_map,
                            koopman_estimator=koopman_estimator,
                            decoder=decoder)
model.fit(X, Y)





