from typing import Optional

from torch import Tensor
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from typing import Tuple

from snowfall.models import AcousticModel
from snowfall.models.modules import ConvModule, Normalize, SlowBatchnorm
from snowfall.training.diagnostics import measure_weight_norms


class Foo(AcousticModel):
    """
    Args:
        num_features (int): Number of input features, e.g. 80
        num_classes (int): Number of output classes, e.g. 103
    """

    def __init__(self,
                 num_features: int,
                 num_classes: int,
                 dim: int = 256,
                 bottleneck_dim: int = 64,
                 groups: int = 4,
                 kernel_size: int = 3,
                 dropout: float = 0.1,
                 num_layers: Tuple[int] = (5, 12),
                 hidden_dim: int = 512,
                 initial_batchnorm_scale=0.2) -> None:
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.subsampling_factor = 4

        self.input_layers = nn.Sequential(
            nn.Conv1d(in_channels=num_features,
                      out_channels=dim,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=dim, affine=False),
            nn.Conv1d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm1d(num_features=dim, affine=False),
            nn.Conv1d(in_channels=dim,
                      out_channels=dim,
                      kernel_size=3,
                      stride=2,
                      padding=1), nn.ReLU(inplace=True))

        conv_module_args = { 'idim':dim, 'odim':dim,
                             'hidden_dim':hidden_dim,
                             'bottleneck_dim': bottleneck_dim,
                             'groups': groups,
                             'kernel_size': kernel_size,
                             'dropout': dropout,
                             'initial_batchnorm_scale': initial_batchnorm_scale }


        self.layers_before_subsample2 = nn.Sequential(* [ ConvModule(**conv_module_args) for
                                                        _ in range(num_layers[0]) ])
        self.subsample2 = ConvModule(stride=2, **conv_module_args)
        self.layers_after_subsample2 = nn.Sequential(* [ ConvModule(**conv_module_args) for
                                                        _ in range(num_layers[1]) ])

        self.final_conv1d = nn.Conv1d(dim, num_classes, stride=1, kernel_size=1, bias=True)


    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (torch.Tensor): Tensor of dimension (batch_size, num_features, input_length).

        Returns:
            Tensor: Predictor tensor of dimension (batch_size, number_of_classes, input_length).
        """
        x = self.input_layers(x)
        x = self.layers_before_subsample2(x)
        x = self.subsample2(x)
        x = self.layers_after_subsample2(x)
        x = self.final_conv1d(x)
        x = nn.functional.log_softmax(x, dim=1)
        return x


    def write_tensorboard_diagnostics(
            self,
            tb_writer: SummaryWriter,
            global_step: Optional[int] = None
    ):
        tb_writer.add_scalars(
            'train/weight_l2_norms',
            measure_weight_norms(self, norm='l2'),
            global_step=global_step
        )
        tb_writer.add_scalars(
            'train/weight_max_norms',
            measure_weight_norms(self, norm='linf'),
            global_step=global_step
        )
