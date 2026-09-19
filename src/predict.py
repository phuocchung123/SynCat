import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from reaction_data import get_graph_data
from data import GraphDataset
from utils import collate_reaction_graphs
from model import model
from typing import List, Tuple

# Architecture of checkpoints saved before the model settings were stored in
# them ("model_config"); these were all trained with the default settings.
LEGACY_CONFIG = {"num_layer": 3, "emb_dim": 384, "drop_ratio": 0.1}


def predict(
    rsmi_lst: List[str],
    model_path: str = "../Data/model/",
    model_name: str = "model_yield",
    device: int = 0,
) -> List[Tuple[torch.Tensor, List, List, List]]:
    """
    Run inference on a list of reaction SMILES strings.

    Each element in `rsmi_lst` is expected to be of the form "reactant_smiles>>product_smiles".
    The model architecture (GNN layers, embedding size, attention layers and
    heads) is rebuilt from the settings stored in the checkpoint, so any model
    trained with `main_finetune.py` can be loaded without repeating them.

    Parameters:
        rsmi_lst: List of reaction SMILES strings, each formatted as "reactant>>product".
        model_path: Directory prefix where the model checkpoint `.pt` file is stored.
        model_name: Checkpoint file name, with or without the ".pt" extension
            (e.g. "model_yield" or "model_yield.pt", as given to `--model_name`).
        device: GPU index to use if CUDA is available; falls back to CPU otherwise.

    Returns:
        A list of predictions, one per batch. Each prediction is expected to be a tuple
        (pred, emb) as returned by the model during inference, where
        `pred` holds the predicted yield values of shape [batch_size].

    """

    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_lst])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_lst])

    rmol, pmol, reaction = get_graph_data(rsmi_lst, rmol_max_cnt, pmol_max_cnt)
    dataset = GraphDataset(
        save_path=None,
        rmol=rmol,
        pmol=pmol,
        reaction=reaction,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(np.min([len(dataset), 8])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,
    )

    node_dim = dataset.rmol_node_attr[0].shape[1]
    edge_dim = dataset.rmol_edge_attr[0].shape[1]

    device = (
        torch.device(f"cuda:{device}")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if not model_name.endswith(".pt"):
        model_name += ".pt"
    checkpoint = torch.load(
        model_path + model_name, map_location=device, weights_only=False
    )
    config = checkpoint.get("model_config")
    if config is None:
        config = dict(LEGACY_CONFIG, node_in_feats=node_dim, edge_in_feats=edge_dim)
    elif (config["node_in_feats"], config["edge_in_feats"]) != (node_dim, edge_dim):
        raise ValueError(
            "The checkpoint expects node/edge features of size %d/%d, but the "
            "input reactions were featurized to %d/%d"
            % (config["node_in_feats"], config["edge_in_feats"], node_dim, edge_dim)
        )
    net = model.from_config(config).to(device)
    net.load_state_dict(checkpoint["model_state_dict"])

    net.eval()
    prediction_list = []
    with torch.no_grad():
        for batchdata in tqdm(loader, desc="Prediction"):
            rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            pmol = [
                b.to(device)
                for b in batchdata[rmol_max_cnt: rmol_max_cnt + pmol_max_cnt]
            ]
            r_dummy = batchdata[-4]
            p_dummy = batchdata[-3]

            prediction = net(
                rmol, pmol, r_dummy, p_dummy, device
            )  # prediction: pred, emb
            prediction_list.append(prediction)
    return prediction_list
