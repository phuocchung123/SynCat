import numpy as np
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from reaction_data import get_graph_data
from data import GraphDataset
from utils import collate_reaction_graphs
from model import model
from typing import List, Tuple, Any


def predict(rsmi_lst: List[str], 
            model_path: str = '../Data/model/',
            model_name: str = 'model_tpl', 
            device: int = 0
            )->List[Tuple[torch.Tensor, List, List, List]]:
    """
    Load SynCat model and run inference on a list of reaction SMILES strings.

    Each element in `rsmi_lst` is expected to be of the form "reactant_smiles>>product_smiles".
    The model choices currently supported are 'model_schneider' and 'model_tpl', which
    determine output dimensionality and architecture hyperparameters.

    Parameters:
        rsmi_lst: List of reaction SMILES strings, each formatted as "reactant>>product".
        model_path: Directory prefix where the model checkpoint `.pt` file is stored.
        model_name: Name of the model to load; must be either 'model_schneider' or 'model_tpl'.
        device: GPU index to use if CUDA is available; falls back to CPU otherwise.

    Returns:
        A list of predictions, one per batch. Each prediction is expected to be a tuple
        (pred, att_r, att_p, emb) as returned by the model during inference

    """
    
    if model_name =='model_schneider':
        out_dim = 50
        layer = 2
        emb_dim = 256
    elif model_name == 'model_tpl':
        out_dim = 1000
        layer = 3
        emb_dim = 384 
    else:
        raise ValueError(f"This model does not exist. Please check the model's name or its appearance again.")
    
    
    rmol_max_cnt = np.max([smi.split(">>")[0].count(".") + 1 for smi in rsmi_lst])
    pmol_max_cnt = np.max([smi.split(">>")[1].count(".") + 1 for smi in rsmi_lst])
    
    rmol, pmol, reaction = get_graph_data(rsmi_lst, rmol_max_cnt, pmol_max_cnt)
    dataset = GraphDataset(
        save_path = None,
        rmol = rmol,
        pmol = pmol,
        reaction = reaction,
    )
    loader = DataLoader(
        dataset=dataset,
        batch_size=int(np.min([len(dataset), 8])),
        shuffle=False,
        collate_fn=collate_reaction_graphs,)
    
    node_dim = dataset.rmol_node_attr[0].shape[1] 
    edge_dim = dataset.rmol_edge_attr[0].shape[1]
    
    device = (torch.device(f"cuda:{device}") if torch.cuda.is_available()
            else torch.device("cpu"))
    net = model(node_dim, edge_dim, out_dim, layer, emb_dim, 0.1).to(device)
    checkpoint = torch.load(model_path + model_name + ".pt", map_location = device, weights_only = False)
    net.load_state_dict(checkpoint['model_state_dict'])
    
    net.eval()
    prediction_list = []
    with torch.no_grad():
        for batchdata in tqdm(loader, desc='Prediction'):
            rmol = [b.to(device) for b in batchdata[:rmol_max_cnt]]
            pmol = [b.to(device) for b in batchdata[rmol_max_cnt:rmol_max_cnt+pmol_max_cnt]]
            r_dummy = batchdata[-4]
            p_dummy = batchdata[-3]

            prediction = net(rmol, pmol, r_dummy, p_dummy, device) # prediction: pred, att_r, att_p, emb 
            prediction_list.append(prediction)
    return prediction_list
                
    
    