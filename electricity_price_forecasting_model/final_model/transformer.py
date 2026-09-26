"""Abstract sequence-model extension point for a future Transformer design.

The shared pipeline already prepares historical windows. A concrete subclass
can compose nn.Module components from the layers package and implement
_build_network(input_size), returning one scalar per window.
"""

from pipeline.neural import TorchForecastModel


class TransformerModel(TorchForecastModel):
    """Define the training interface without choosing a Transformer architecture.

    Keep this class out of the runnable registry until a concrete architecture
    has been designed and tested. Subclasses can set their own sequence_length
    and implement _build_network(input_size) using reusable layers.
    """

    model_name = "Transformer"
    requires_sequence = True
    sequence_length = 36
