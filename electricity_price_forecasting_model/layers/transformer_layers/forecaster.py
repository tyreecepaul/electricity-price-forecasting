from electricity_price_forecasting_model.layers.transformer_layers.transformerLayers import DecoderTransformer

from __future__ import annotations

from torch import nn

class Forecaster(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward=2048, n_epochs= 100, batch_size=32, lr_rate=0.001, random_seed=42):
        super(Forecaster, self).__init__()
        self.input_layer = nn.Linear(1, d_model)  # Assuming input is a single value
        self.transformer = DecoderTransformer(d_model, nhead, num_layers, dim_feedforward)
        self.output_layer = nn.Linear(d_model, 1)  # Assuming output is a single value

    def forward(self, x, mask=None):
        x = self.input_layer(x) # get rid of the project here?
                                # need input embeddings somehow
        x = self.transformer(x, mask)
        return self.output_layer(x)





        