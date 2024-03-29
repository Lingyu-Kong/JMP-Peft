import ll.nn
from ll import TypedConfig


class AdapterLayerConfig(TypedConfig):
    in_dim: int
    bottleneck_dim: int
    out_dim: int
    nonlinearity: ll.nn.NonlinearityConfig
    bias: bool = True
    dropout: float | None = None
    residual: bool = False

    def create_module(self):
        return ll.nn.MLP(
            [self.in_dim, self.bottleneck_dim, self.out_dim],
            activation=self.nonlinearity,
            bias=self.bias,
            dropout=self.dropout,
            residual=self.residual,
        )


class DLoraOutputBlockConfig(TypedConfig):
    pass


class DLoraConfig(TypedConfig):
    output_block: DLoraOutputBlockConfig = DLoraOutputBlockConfig()
