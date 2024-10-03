import sys
import pathlib

root_dir = str(pathlib.Path(__file__).resolve().parents[3])
sys.path.append(root_dir)
from tqdm import tqdm
from KGGraph.KGGProcessor.pretrain_dataset import molgraph_to_graph_data


def group_node_rep(node_rep, batch_size, num_part):
    """
    Groups the node representations based on the batch size and number of partitions.

    Args:
        node_rep (list): The list of node representations.
        batch_size (int): The size of the batch.
        num_part (list): The list of numbers of partitions for each batch.

    Returns:
        tuple: A tuple containing two lists - group and super_group.
            - group (list): The grouped node representations.
            - super_group (list): The super group node representations.

    Examples:
        >>> node_rep = [1, 2, 3, 4, 5, 6]
        >>> batch_size = 2
        >>> num_part = [[2, 1], [1, 1]]
        >>> group_node_rep(node_rep, batch_size, num_part)
        ([[1, 2], [3]], [4, 6])
    """

    group = []
    super_group = []
    # print('num_part', num_part)
    count = 0
    for i in range(batch_size):
        num_atom = num_part[i][0]
        num_motif = num_part[i][1]
        num_all = num_atom + num_motif + 1
        group.append(node_rep[count : count + num_atom])
        super_group.append(node_rep[count + num_all - 1])
        count += num_all
    return group, super_group


def train(args, model_list, loader, optimizer_list, device, pretrain_loss, epoch):
    model, model_decoder = model_list

    model.train()
    model_decoder.train()
    # if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = 0, 0, 0, 0, 0, 0

    for step, batch in enumerate(tqdm(loader, desc="KGG Pretraining Step")):
        batch_size = len(batch)

        graph_batch = molgraph_to_graph_data(batch)
        graph_batch = graph_batch.to(device)
        node_rep = model(graph_batch.x, graph_batch.edge_index, graph_batch.edge_attr)
        num_part = graph_batch.num_part
        node_rep, super_node_rep = group_node_rep(node_rep, batch_size, num_part)

        loss = model_decoder(batch, node_rep, super_node_rep)

        optimizer_list.zero_grad()

        loss.backward()

        optimizer_list.step()

        # if_auc += bond_if_auc
        # if_ap += bond_if_ap
        # a_type_acc += atom_type_acc
        # a_num_rmse += atom_num_rmse
        # b_num_rmse += bond_num_rmse

        if (step + 1) % 20 == 0:
            # if_auc = if_auc / 20
            # if_ap = if_ap / 20
            # type_acc = type_acc / 20
            # a_type_acc = a_type_acc / 20
            # a_num_rmse = a_num_rmse / 20
            # b_num_rmse = b_num_rmse / 20

            print("Batch:", step, "loss:", loss.item())
            # if_auc, if_ap, type_acc, a_type_acc, a_num_rmse, b_num_rmse = (
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            #     0,
            # )
        pretrain_loss["loss"][epoch - 1] = loss.item()
    pretrain_loss.to_csv("Data/pretrain_loss.csv")
    return pretrain_loss
