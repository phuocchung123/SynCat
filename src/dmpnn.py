import torch
import torch.nn as nn

from dmpnn_encoder import DMPNNEncoder, MLP


class DMPNN(nn.Module):
    """
    PyG-style D-MPNN model for graph-level molecular property prediction.

    Example:
        model = DMPNN(in_channels=74, edge_channels=14, hidden_channels=300, out_channels=1)
        pred = model(x, edge_index, edge_attr, batch)
    """

    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        hidden_channels: int = 300,
        num_layers: int = 3,
        out_channels: int = 1,
        dropout: float = 0.0,
        pool: str = "mean",
        ffn_num_layers: int = 2,
        task: str = "regression",
    ):
        super().__init__()

        self.task = task

        self.encoder = DMPNNEncoder(
            in_channels=in_channels,
            edge_channels=edge_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            pool=pool,
        )

        self.ffn = MLP(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=ffn_num_layers,
            dropout=dropout,
        )

    def forward(
        self,
        x,
        edge_index=None,
        edge_attr=None,
        batch=None,
        rev_edge_index=None,
    ):
        """
        Supports either:

            model(data)

        or:

            model(
                x, edge_index, edge_attr, batch, rev_edge_index
            )
        """

        if edge_index is None and hasattr(x, "edge_index"):
            data = x
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            batch = getattr(data, "batch", None)
            rev_edge_index = getattr(data, "rev_edge_index", None)
            x = data.x

        graph_hidden = self.encoder(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            rev_edge_index=rev_edge_index,
        )

        out = self.ffn(graph_hidden)

        if self.task == "binary_classification":
            # Return logits during training.
            # Apply sigmoid only for inference.
            return out

        return out