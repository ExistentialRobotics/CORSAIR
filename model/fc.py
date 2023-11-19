import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME


def split_batch(sp):
    feats = []
    batch_size = 1 + sp.C[:, 0].max()
    for i in range(batch_size):
        base_mask = sp.C[:, 0] == i
        feats.append(sp.F[base_mask, :])
    return feats


class FC(nn.Module):
    def __init__(self, dims, num_class=55):
        super(FC, self).__init__()
        self.fc1 = nn.Sequential(
            *[nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

    def _max_pool(self, sparse_tensor):
        coords = sparse_tensor.C
        feats = sparse_tensor.F
        feat_batch = []
        for i in range(coords[:, 0].max().item() + 1):
            mask = coords[:, 0] == i
            feat_batch.append(feats[mask].max(0)[0])
        return torch.stack(feat_batch)

    def max_pool(self, tensor_list):
        feat_batch = []
        for tensor in tensor_list:
            feat_batch.append(tensor.max(0)[0])
        return torch.stack(feat_batch)

    def forward(self, input):
        embedding = self._max_pool(input)
        embedding = self.fc1(embedding)
        return embedding, embedding


class conv1_chamfer(nn.Module):
    def __init__(self, out_channels):
        super(conv1_chamfer, self).__init__()
        self.final = ME.MinkowskiConvolution(
            in_channels=256,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            **{"bias" if ME.__version__ >= "0.5.4" else "has_bias": True},
            dimension=3
        )

    def forward(self, input):
        out = self.final(input)
        return out


class conv1_fc_chamfer(nn.Module):
    def __init__(self, conv_channels, linear1_dim, linear2_dim):
        super(conv1_fc_chamfer, self).__init__()
        self.final = conv1_chamfer(conv_channels)
        self.fc1 = nn.Linear(conv_channels, linear1_dim)
        self.fc2 = nn.Linear(linear1_dim, linear2_dim)
        self.bn1 = nn.BatchNorm1d(linear1_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        feats = self.final(input)
        x = self.relu(self.bn1(self.fc1(feats.F)))
        out = self.fc2(x)
        #         x = self.fc2(self.fc1(feats.F))
        #         out = self.relu(self.bn1(x))
        output = ME.SparseTensor(feats=out, coords=feats.C).to("cuda")
        return output


class max_embedding(nn.Module):
    def __init__(self, feat_dim, linear1_dim, linear2_dim):
        super(max_embedding, self).__init__()
        # self.final = conv1_chamfer(conv_channels)
        self.fc1 = nn.Linear(feat_dim, linear1_dim)
        self.fc2 = nn.Linear(linear1_dim, linear2_dim)
        self.bn1 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, input):
        feats = split_batch(input)
        mean_feats = torch.stack([feat.max(0)[0] for feat in feats])
        x = self.relu(self.bn1(self.fc1(mean_feats)))
        out = self.fc2(x)
        return out


class conv1_max_embedding(nn.Module):
    def __init__(self, conv_channels, linear1_dim, linear2_dim):
        super(conv1_max_embedding, self).__init__()
        self.final = conv1_chamfer(conv_channels)
        self.fc1 = nn.Linear(conv_channels, linear1_dim)
        self.fc2 = nn.Linear(linear1_dim, linear2_dim)
        self.bn1 = nn.BatchNorm1d(linear1_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        feats = split_batch(self.final(input))
        mean_feats = torch.stack([feat.max(0)[0] for feat in feats])
        x = self.relu(self.bn1(self.fc1(mean_feats)))
        out = self.fc2(x)
        return out


class identity(nn.Module):
    def __init__(self):
        super(identity, self).__init__()

    def forward(self, input):
        return input


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=32, dim=256, alpha=100.0, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.fc = nn.Linear(num_clusters * dim, 1024)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(-self.alpha * self.centroids.norm(dim=1))
        print("weight", self.conv.weight.size())
        print("bias", self.conv.bias.size())

    def forward(self, x):
        coords = x.C
        feat = x.F
        x = x.F
        x = x.view(1, x.size()[1], x.size()[0], 1)

        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        # print(soft_assign.shape)

        x_flatten = x.view(N, C, -1)
        batch_size = coords[:, 0].max().item() + 1
        vlad_full = torch.rand(batch_size, self.dim * self.num_clusters).cuda()
        for i in range(batch_size):
            mask = coords[:, 0] == i
            soft = F.softmax(soft_assign[:, :, mask], dim=1)
            x_point = x_flatten[:, :, mask]
            residual = x_point.expand(self.num_clusters, -1, -1, -1).permute(
                1, 0, 2, 3
            ) - self.centroids.expand(x_point.size(-1), -1, -1).permute(
                1, 2, 0
            ).unsqueeze(
                0
            )
            # print(residual)
            residual *= soft.unsqueeze(2)
            vlad = residual.sum(dim=-1)
            vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
            vlad = vlad.view(x.size(0), -1)  # flatten
            # vlad = F.normalize(vlad, p=2, dim=1)
            vlad_full[i, :] = vlad

        final_feature = self.fc(vlad_full)
        final_feature = F.normalize(final_feature, dim=1)

        # print(final_feature)
        return final_feature  # ,vlad_full.view(batch_size,self.num_clusters,self.dim)


class NetVLADLoupe(nn.Module):
    def __init__(
        self,
        feature_size=256,
        max_samples=4000,
        cluster_size=64,
        output_dim=1024,
        gating=False,
        add_batch_norm=True,
        is_training=True,
    ):
        super(NetVLADLoupe, self).__init__()
        self.feature_size = feature_size
        self.max_samples = max_samples
        self.output_dim = output_dim
        self.is_training = is_training
        self.gating = gating
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        self.softmax = nn.Softmax(dim=-1)
        self.cluster_weights = nn.Parameter(
            torch.randn(feature_size, cluster_size) * 1 / math.sqrt(feature_size)
        )
        self.cluster_weights2 = nn.Parameter(
            torch.randn(1, feature_size, cluster_size) * 1 / math.sqrt(feature_size)
        )
        self.hidden1_weights = nn.Parameter(
            torch.randn(cluster_size * feature_size, output_dim)
            * 1
            / math.sqrt(feature_size)
        )

        if add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(cluster_size)
        else:
            self.cluster_biases = nn.Parameter(
                torch.randn(cluster_size) * 1 / math.sqrt(feature_size)
            )
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(output_dim)

        if gating:
            self.context_gating = GatingContext(
                output_dim, add_batch_norm=add_batch_norm
            )

    def forward(self, x):
        coords = x.C
        feat = x.F
        x = x.F

        #             x = x.transpose(1, 3).contiguous()
        #             x = x.view((-1, self.max_samples, self.feature_size))
        x = x.view((1, -1, self.feature_size))
        #         print(x.size())
        activation = torch.matmul(x, self.cluster_weights)
        if self.add_batch_norm:
            # activation = activation.transpose(1,2).contiguous()
            activation = activation.view(-1, self.cluster_size)
            activation = self.bn1(activation)
            activation_full = activation.view(1, -1, self.cluster_size)
            # activation = activation.transpose(1,2).contiguous()
        else:
            activation_full = activation + self.cluster_biases

        batch_size = coords[:, 0].max().item() + 1
        vlad_full = torch.rand(batch_size, self.feature_size * self.cluster_size).cuda()
        for i in range(batch_size):
            mask = coords[:, 0] == i

            activation = activation_full[:, mask, :]
            activation = self.softmax(activation)
            activation = activation.view((1, -1, self.cluster_size))
            a_sum = activation.sum(-2, keepdim=True)
            a = a_sum * self.cluster_weights2

            activation = torch.transpose(activation, 2, 1)
            x_new = x[:, mask, :]
            x_new = x_new.view((1, -1, self.feature_size))
            vlad = torch.matmul(activation, x_new)
            vlad = torch.transpose(vlad, 2, 1)
            vlad = vlad - a
            vlad = F.normalize(vlad, dim=1, p=2)
            vlad = vlad.reshape((-1, self.cluster_size * self.feature_size))
            vlad = F.normalize(vlad, dim=1, p=2)
            vlad_full[i, :] = vlad

        vlad = torch.matmul(vlad_full, self.hidden1_weights)

        vlad = self.bn2(vlad)

        if self.gating:
            vlad = self.context_gating(vlad)

        return vlad


class GatingContext(nn.Module):
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)

        activation = x * gates

        return activation
