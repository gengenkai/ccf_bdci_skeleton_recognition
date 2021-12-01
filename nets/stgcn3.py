import numpy as np
import torch
import torch.nn as nn
from mmaction.models.builder import BACKBONES
from mmaction.utils import get_root_logger
from mmcv.cnn import constant_init, kaiming_init, normal_init
from mmcv.runner import load_checkpoint
from mmcv.utils import _BatchNorm

from nets.graph import Graph


def zero(x):
    """return zero."""
    return 0


def identity(x):
    """return input itself."""
    return x


class STGCNBlock(nn.Module):
    """Applies a spatial temporal graph convolution over an input graph
    sequence.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (tuple): Size of the temporal convolving kernel and
            graph convolving kernel
        stride (int, optional): Stride of the temporal convolution. Default: 1
        dropout (int, optional): Dropout rate of the final output. Default: 0
        residual (bool, optional): If ``True``, applies a residual mechanism.
            Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Outpu graph sequence in :math:`(N, out_channels, T_{out},
            V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V,
            V)` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size,
                 stride=1,
                 dropout=0,
                 residual=True):
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, A,
                                         kernel_size[1])
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1),
                      (stride, 1), padding), nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True))

        if not residual:
            self.residual = zero

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = identity

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels,
                          kernel_size=1,
                          stride=(stride, 1)), nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Defines the computation performed at every call."""
        res = self.residual(x)
        x = self.gcn(x)

        x = self.tcn(x) + res

        return self.relu(x)


class ConvTemporalGraphical(nn.Module):
    """The basic module for applying a graph convolution.

    Args:
        in_channels (int): Number of channels in the input sequence data
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int): Size of the graph convolving kernel
        t_kernel_size (int): Size of the temporal convolving kernel
        t_stride (int, optional): Stride of the temporal convolution.
            Default: 1
        t_padding (int, optional): Temporal zero-padding added to both sides
            of the input. Default: 0
        t_dilation (int, optional): Spacing between temporal kernel elements.
            Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Default: ``True``

    Shape:
        - Input[0]: Input graph sequence in :math:`(N, in_channels, T_{in}, V)`
            format
        - Input[1]: Input graph adjacency matrix in :math:`(K, V, V)` format
        - Output[0]: Output graph sequence in :math:`(N, out_channels, T_{out}
            , V)` format
        - Output[1]: Graph adjacency matrix for output data in :math:`(K, V, V)
            ` format

        where
            :math:`N` is a batch size,
            :math:`K` is the spatial kernel size, as :math:`K == kernel_size[1]
                `,
            :math:`T_{in}/T_{out}` is a length of input/output sequence,
            :math:`V` is the number of graph nodes.
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 A,
                 kernel_size,
                 t_kernel_size=1,
                 t_stride=1,
                 t_padding=0,
                 t_dilation=1,
                 bias=True):
        super().__init__()

        self.kernel_size = kernel_size
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))

        self.num_subset = 3
        inter_channels = out_channels // 4
        self.inter_c = inter_channels
        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1),
                                      nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        # adaptive
        self.alpha = nn.Parameter(torch.zeros(1))
        self.sigmoid = nn.Sigmoid()

        # spatial attention
        num_jpts = A.shape[-1]
        ker_jpt = num_jpts - 1 if not num_jpts % 2 else num_jpts
        pad = (ker_jpt - 1) // 2
        self.conv_sa = nn.Conv1d(out_channels, 1, ker_jpt, padding=pad)
        nn.init.xavier_normal_(self.conv_sa.weight)
        nn.init.constant_(self.conv_sa.bias, 0)

        # temporal attention
        self.conv_ta = nn.Conv1d(out_channels, 1, 9, padding=4)
        nn.init.constant_(self.conv_ta.weight, 0)
        nn.init.constant_(self.conv_ta.bias, 0)

        # channel attention
        rr = 2
        self.fc1c = nn.Linear(out_channels, out_channels // rr)
        self.fc2c = nn.Linear(out_channels // rr, out_channels)
        nn.init.kaiming_normal_(self.fc1c.weight)
        nn.init.constant_(self.fc1c.bias, 0)
        nn.init.constant_(self.fc2c.weight, 0)
        nn.init.constant_(self.fc2c.bias, 0)

    def forward(self, x):
        # add A B C
        """Defines the computation performed at every call."""

        N, C, T, V = x.size()
        # A = adj_mat + self.PA
        A = self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).contiguous().view(
                N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).view(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 * self.alpha + A[i]
            A2 = x.view(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).view(N, C, T, V))
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)

        y = self.relu(y)

        # spatial attention
        se = y.mean(-2)  # N C V
        se1 = self.sigmoid(self.conv_sa(se))
        y = y * se1.unsqueeze(-2) + y

        # temporal attention
        se = y.mean(-1)
        se1 = self.sigmoid(self.conv_ta(se))
        y = y * se1.unsqueeze(-1) + y

        # channel attention
        se = y.mean(-1).mean(-1)
        se1 = self.relu(self.fc1c(se))
        se2 = self.sigmoid(self.fc2c(se1))
        y = y * se2.unsqueeze(-1).unsqueeze(-1) + y

        return y


@BACKBONES.register_module()
class STGCN3(nn.Module):
    """Backbone of AAGCN(the graph, tcn, init methods are a bit different from
    the original repo).

    Args:
        in_channels (int): Number of channels in the input data.
        graph_cfg (dict): The arguments for building the graph.
        edge_importance_weighting (bool): If ``True``, adds a learnable
            importance weighting to the edges of the graph. Default: True.
        data_bn (bool): If 'True', adds data normalization to the inputs.
            Default: True.
        pretrained (str | None): Name of pretrained model.
        **kwargs (optional): Other parameters for graph convolution units.

    Shape:
        - Input: :math:`(N, in_channels, T_{in}, V_{in}, M_{in})`
        - Output: :math:`(N, num_class)` where
            :math:`N` is a batch size,
            :math:`T_{in}` is a length of input sequence,
            :math:`V_{in}` is the number of graph nodes,
            :math:`M_{in}` is the number of instance in a frame.
    """
    def __init__(self,
                 in_channels,
                 graph_cfg,
                 data_bn=True,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        # load graph
        self.graph = Graph(**graph_cfg)
        A = self.graph.A

        num_joint = A.shape[-1]

        # build networks
        spatial_kernel_size = 3
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels *
                                      num_joint) if data_bn else identity

        kwargs0 = {k: v for k, v in kwargs.items() if k != 'dropout'}
        self.st_gcn_networks = nn.ModuleList((
            STGCNBlock(in_channels,
                       64,
                       A,
                       kernel_size,
                       1,
                       residual=False,
                       **kwargs0),
            STGCNBlock(64, 64, A, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, A, kernel_size, 1, **kwargs),
            STGCNBlock(64, 64, A, kernel_size, 1, **kwargs),
            STGCNBlock(64, 128, A, kernel_size, 2, **kwargs),
            STGCNBlock(128, 128, A, kernel_size, 1, **kwargs),
            STGCNBlock(128, 128, A, kernel_size, 1, **kwargs),
            STGCNBlock(128, 256, A, kernel_size, 2, **kwargs),
            STGCNBlock(256, 256, A, kernel_size, 1, **kwargs),
            STGCNBlock(256, 256, A, kernel_size, 1, **kwargs),
        ))

        self.pretrained = pretrained

    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, nn.Linear):
                    normal_init(m)
                elif isinstance(m, _BatchNorm):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        # data normalization
        x = x.float()
        n, c, t, v, m = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(n * m, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, m, v, c, t)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(n * m, c, t, v)

        for gcn in self.st_gcn_networks:
            x = gcn(x)

        return x
