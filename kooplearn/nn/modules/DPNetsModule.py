from .KoopmanDNNModule import KoopmanDNNModule


class DPNetsModule(KoopmanDNNModule):
    def base_step(self, batch, batch_idx):
        # dimensions convention (..., channels, temporal_dim)
        # here we assume that the same model encodes input and output
        data_x = {'x_value': batch['x_value']}
        data_y = {'x_value': batch['y_value']}
        model_output_x = self(data_x)
        x_encoded = model_output_x['x_encoded']
        model_output_y = self(data_y)
        y_encoded = model_output_y['x_encoded']
        loss = self.loss_fn(x_encoded, y_encoded)
        outputs = {
            'loss': loss,
        }
        return outputs
    