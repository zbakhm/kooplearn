from kooplearn.encoder_decoder.decoder.Decoder import Decoder


class CustomDecoder(Decoder):
    def __init__(self, decoder_fn):
        super().__init__()
        self.decoder_fn = decoder_fn

    def forward(self, x):
        return self.decoder_fn(x)