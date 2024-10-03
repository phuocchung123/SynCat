import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import random
from typing import Tuple, Optional
from rdkit import Chem
from rdkit.Chem import Draw
import io


def tree_vis(
    G: nx.DiGraph,
    node_label_attr: str = "ismiles",
    edge_label_attr: str = "label",
    edge_labels: bool = False,
    show_clusters: bool = False,
    seed: Optional[int] = 42,
    figsize: Tuple[int, int] = (10, 8),
    node_size: int = 500,
    font_size: int = 12,
    title: str = "Graph Visualization",
    colormap: ScalarMappable = plt.cm.coolwarm,
) -> plt.Figure:
    """
    Visualize a directed graph with specific features, reorienting edges based on edge labels.

    Args:
    - G (nx.DiGraph): A NetworkX directed graph.
    - node_label_attr (str): The node attribute key for labels.
    - edge_label_attr (str): The edge attribute key for labels.
    - edge_labels (bool, optional): If True, edge labels/attributes are displayed.
    - show_clusters (bool, optional): If True, node labels are displayed with clusters.
    - seed (int, optional): Seed for random state for reproducibility.
    - figsize (tuple): Figure size for the plot.
    - node_size (int): Size of the nodes.
    - font_size (int): Font size for text.
    - title (str): Title of the plot.
    - colormap (ScalarMappable): Colormap used for node coloring.

    Returns:
    - fig (plt.Figure): The generated plot.
    """
    if seed is not None:
        random.seed(seed)

    fig = plt.figure(figsize=figsize)
    pos = nx.spring_layout(G, seed=seed)

    node_labels = nx.get_node_attributes(G, node_label_attr)

    if node_labels:
        unique_labels = set(node_labels.values())
        color_map = {
            label: colormap(i / len(unique_labels))
            for i, label in enumerate(unique_labels)
        }
        nx.draw_networkx_nodes(
            G,
            pos,
            node_color=[color_map[node_labels[node]] for node in G.nodes],
            node_size=node_size,
        )
    else:
        nx.draw_networkx_nodes(G, pos, node_size=node_size)

    if show_clusters:
        labels_to_show = {
            node: label for node, label in node_labels.items() if label in unique_labels
        }
        nx.draw_networkx_labels(G, pos, labels=labels_to_show, font_size=font_size)
    else:
        nx.draw_networkx_labels(G, pos, font_size=font_size)

    UG = G.to_undirected()
    edge_labels_data = nx.get_edge_attributes(G, edge_label_attr)
    edges_in_order = sorted(
        UG.edges(data=True),
        key=lambda x: edge_labels_data.get(
            (x[0], x[1]), edge_labels_data.get((x[1], x[0]), float("inf"))
        ),
    )

    DG = nx.DiGraph()
    DG.add_nodes_from(G.nodes(data=True))
    DG.add_edges_from(
        [
            (u, v, d)
            for u, v, d in edges_in_order
            if (u, v) in edge_labels_data or (v, u) in edge_labels_data
        ]
    )

    nx.draw_networkx_edges(DG, pos, arrows=True, arrowstyle="->", arrowsize=10)

    if edge_labels:
        nx.draw_networkx_edge_labels(
            DG, pos, edge_labels=edge_labels_data, font_size=font_size
        )
    plt.axis("off")
    return fig


def vis_compare(
    smiles: str,
    tree_nodes_fig: plt.Figure,
    show_atom_map: bool = True,
    figsize: tuple = (10, 6),
    mol_img_size: tuple = (500, 500),
) -> None:
    """
    Create a 1x2 subplot with a molecule structure from a SMILES string and a tree nodes figure.

    Args:
    - smiles (str): A SMILES string representing a molecule.
    - tree_nodes_fig (plt.Figure): A matplotlib figure representing tree nodes.
    - show_atom_map (bool): If True, shows atom map numbers on the molecule.
    - figsize (tuple): Size of the entire subplot figure.
    - mol_img_size (tuple): Size of the molecule image.
    """
    # Create a subplot layout
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Generate a molecule from the SMILES string
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        if show_atom_map:
            for atom in mol.GetAtoms():
                atom.SetProp("atomLabel", str(atom.GetIdx()))
        # Convert the RDKit drawing to an image
        mol_img = Draw.MolToImage(mol, size=mol_img_size)
        # Display the molecule image in the first subplot
        axes[0].imshow(mol_img)
        axes[0].axis("off")  # Turn off axis for molecule image

    # Display the tree nodes figure in the second subplot
    buf = io.BytesIO()
    tree_nodes_fig.savefig(buf, format="png")
    buf.seek(0)
    img = plt.imread(buf)
    axes[1].imshow(img)
    axes[1].axis("off")  # Turn off axis for tree nodes figure

    # Add a gridline to separate the subfigures
    plt.subplots_adjust(wspace=0.5)

    plt.tight_layout()
    plt.show()
