from .KoopmanDNNModule import KoopmanDNNModule


class DPNetsModule(KoopmanDNNModule):
    def base_step(self, batch, batch_idx):
        # dimensions convention (..., channels, temporal_dim)
        x = batch['x_value']
        y = batch['y_value']
        x_encoded = self.model(x)
        y_encoded = self.model(y)
        loss = self.loss_fn(x_encoded, y_encoded)
        outputs = {
            'loss': loss,
        }
        return outputs
    