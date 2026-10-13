"""Bond-line graph representation for molecules.

The transformation turns a molecular graph into its *line graph*: every
undirected chemical bond becomes one transformed node, and two transformed
nodes are connected when their original bonds share an atom. Isolated atoms
(atoms without any chemical bond) are kept as explicit fallback nodes so no
information is lost.

The features of a transformed (bond) node are derived exclusively from the
source atom and bond features, so the original NPZ schema and preparation
scripts remain untouched:

    f_ij = concat(a_i + a_j, |a_i - a_j|, e_ij, [0])        # a chemical bond
    f_i  = concat(a_i, zeros(atom_dim), zeros(bond_dim), [1])  # isolated atom

The last flag is 0 for a bond node and 1 for an isolated-atom fallback node.
Chemical-bond features therefore live in the transformed *node* features, and
the line-graph edges carry no features at all.
"""

import torch
from torch_geometric.data import Data

# Bump when the layout of the transformed features changes so that stale
# checkpoints are rejected instead of silently loaded.
LINE_FEATURE_VERSION = 1

# Flag values stored in the last feature column of a transformed node.
LINE_NODE_BOND = 0
LINE_NODE_ISOLATED = 1


def bond_line_node_dim(atom_dim: int, bond_dim: int) -> int:
    """
    Width of a transformed bond-line node feature vector.

    Parameters
    ----------
    atom_dim : int
        Source atom feature width (e.g. 155).
    bond_dim : int
        Source chemical-bond feature width (e.g. 9).

    Returns
    -------
    int
        ``2 * atom_dim + bond_dim + 1`` (e.g. 320).
    """
    return 2 * int(atom_dim) + int(bond_dim) + 1


class BondLineData(Data):
    """
    PyG ``Data`` subclass for the bond-line representation.

    Only two extra batching rules are needed: ``rev_edge`` stores, for every
    directed line-graph edge, the position of the opposite directed edge
    (edge-on-edge), so it is offset by the number of edges during batching and
    concatenated along dimension 0.
    """

    def __inc__(self, key, value, *args, **kwargs):
        if key == "rev_edge":
            return self.num_edges
        return super().__inc__(key, value, *args, **kwargs)

    def __cat_dim__(self, key, value, *args, **kwargs):
        if key == "rev_edge":
            return 0
        return super().__cat_dim__(key, value, *args, **kwargs)


def _empty_placeholder(x: torch.Tensor, line_dim: int) -> BondLineData:
    """Canonical masked placeholder for a padding (dummy) molecule slot."""
    return BondLineData(
        x=x.new_zeros((1, line_dim)),
        edge_index=torch.empty((2, 0), dtype=torch.long, device=x.device),
        rev_edge=torch.empty((0,), dtype=torch.long, device=x.device),
    )


def _dedupe_chemical_bonds(edge_index, edge_attr):
    """
    Group the directed chemical edges into undirected bonds.

    The source featurizer writes all forward edges first and all reverse edges
    afterwards, but the result must not depend on that ordering, so bonds are
    grouped by their ``(min, max)`` endpoint pair. Both directions must be
    present and carry identical attributes.

    Returns
    -------
    bond_id : dict
        ``(u, v) -> transformed node index`` in first-appearance order.
    bond_attr : dict
        ``(u, v) -> chemical-bond feature row``.
    """
    src, dst = edge_index[0], edge_index[1]
    bond_id, bond_attr, bond_dirs = {}, {}, {}
    for e in range(edge_index.shape[1]):
        u, v = int(src[e]), int(dst[e])
        if u == v:
            raise ValueError(
                "chemical bond self-loop at atom %d is not supported by the "
                "bond-line transformation" % u
            )
        key = (min(u, v), max(u, v))
        attr = edge_attr[e]
        if key not in bond_id:
            bond_id[key] = len(bond_id)
            bond_attr[key] = attr
            bond_dirs[key] = {(u, v)}
        else:
            if not torch.equal(attr, bond_attr[key]):
                raise ValueError(
                    "chemical bond %s has mismatched features on its two "
                    "directions" % (key,)
                )
            bond_dirs[key].add((u, v))

    for (u, v), seen in bond_dirs.items():
        if (u, v) not in seen or (v, u) not in seen:
            raise ValueError(
                "chemical bond %s is missing a reverse directed edge; the "
                "source graph is not undirected" % ((u, v),)
            )
    return bond_id, bond_attr


def _node_features(x, bond_id, bond_attr, atom_dim, bond_dim, line_dim):
    """Build the transformed node features: bond nodes then isolated atoms."""
    zero_atom = x.new_zeros(atom_dim)
    zero_bond = x.new_zeros(bond_dim)
    atom_has_bond = set()
    features = []
    for key in sorted(bond_id, key=lambda k: bond_id[k]):  # first-appearance order
        u, v = key
        atom_has_bond.add(u)
        atom_has_bond.add(v)
        features.append(
            torch.cat(
                [
                    x[u] + x[v],
                    torch.abs(x[u] - x[v]),
                    bond_attr[key],
                    x.new_zeros(1),
                ]
            )
        )
    for i in range(int(x.shape[0])):
        if i not in atom_has_bond:
            features.append(torch.cat([x[i], zero_atom, zero_bond, x.new_ones(1)]))
    if features:
        return torch.stack(features, dim=0)
    return x.new_zeros((0, line_dim))


def _line_adjacency(atom_bonds):
    """
    Directed line-graph edges: every unordered pair of bonds sharing an atom,
    emitted in both directions.
    """
    directed = []
    seen = set()
    for incident in atom_bonds:
        for a in range(len(incident)):
            for b in range(a + 1, len(incident)):
                for pair in (
                    (incident[a], incident[b]),
                    (incident[b], incident[a]),
                ):
                    if pair not in seen:
                        seen.add(pair)
                        directed.append(pair)
    return directed


def _reverse_edges(directed):
    """Edge-on-edge map from every directed line edge to its opposite."""
    position = {pair: idx for idx, pair in enumerate(directed)}
    rev = []
    for a, b in directed:
        opposite = position.get((b, a))
        if opposite is None:
            raise ValueError(
                "bond-line adjacency (%d, %d) has no reverse; internal error"
                % (a, b)
            )
        rev.append(opposite)
    return rev


def to_bond_line(data: Data, is_real: bool = True) -> BondLineData:
    """
    Deterministically transform one molecule graph into its bond-line graph.

    Parameters
    ----------
    data : torch_geometric.data.Data
        Source molecule with float32 ``x`` of width ``atom_dim``, ``edge_index``
        of shape [2, E] (every chemical bond stored in both directions) and
        ``edge_attr`` of width ``bond_dim``.
    is_real : bool, optional
        ``False`` for a masked padding slot, which becomes a canonical
        one-node placeholder (default is True).

    Returns
    -------
    BondLineData
        Transformed graph with ``x`` of width ``2*atom_dim + bond_dim + 1``,
        topology-only ``edge_index`` and the ``rev_edge`` edge-on-edge map.

    Raises
    ------
    ValueError
        If the source graph is inconsistent (a non-reciprocal chemical bond, a
        chemical self-loop, or differing attributes on the two directions of a
        bond).
    """
    x = data.x
    atom_dim = int(x.shape[1])
    bond_dim = int(data.edge_attr.shape[1]) if data.edge_attr.dim() == 2 else 0
    line_dim = bond_line_node_dim(atom_dim, bond_dim)

    if not is_real:
        return _empty_placeholder(x, line_dim)

    edge_index, edge_attr = data.edge_index, data.edge_attr
    n_atoms = int(x.shape[0])

    bond_id, bond_attr = _dedupe_chemical_bonds(edge_index, edge_attr)
    line_x = _node_features(x, bond_id, bond_attr, atom_dim, bond_dim, line_dim)

    atom_bonds = [[] for _ in range(n_atoms)]
    for (u, v), bid in bond_id.items():
        atom_bonds[u].append(bid)
        atom_bonds[v].append(bid)
    directed = _line_adjacency(atom_bonds)

    if directed:
        line_edge_index = torch.tensor(
            directed, dtype=torch.long, device=x.device
        ).t().contiguous()
        rev_edge = torch.tensor(
            _reverse_edges(directed), dtype=torch.long, device=x.device
        )
    else:
        line_edge_index = torch.empty((2, 0), dtype=torch.long, device=x.device)
        rev_edge = torch.empty((0,), dtype=torch.long, device=x.device)

    return BondLineData(x=line_x, edge_index=line_edge_index, rev_edge=rev_edge)
