"""
Explicit hierarchical packing for the four-level reaction-yield model.

The legacy data flow collates one :class:`torch_geometric.data.Batch` per
molecule slot. Each of those batches holds every reaction's graph for that slot,
including a one-atom padding graph wherever the slot is empty. The hierarchical
model instead needs a single flat atom graph containing only the real supplied
molecules, together with the membership indices that map

    atoms -> molecules -> side groups -> reactions.

The packing here never mutates the cached features or the incoming graph
objects: every tensor is produced by indexing/copying. Padding atoms and every
edge that touches one are dropped before the atom encoder and the GINE passes
see them, so padding stays irrelevant even if its features or edges are
non-zero.

Group-node order is fixed to

    reaction 0 Reactants, reaction 0 Products,
    reaction 1 Reactants, reaction 1 Products, ...

which is what ``side2rxn = arange(B).repeat_interleave(2)`` and
``mol2side = 2 * mol2rxn + mol_side`` encode. A molecule identity is always the
(reaction, side, slot) occurrence, never the slot number alone, so repeated
identical SMILES keep independent states.
"""

from typing import List, NamedTuple

import torch

REACTANTS = 0
PRODUCTS = 1


class HierarchicalBatch(NamedTuple):
    """
    Packed real graphs plus the membership links of the four-level hierarchy.

    Attributes
    ----------
    x : torch.Tensor
        Atom features of the real molecules, shape [N, node_in_feats].
    edge_index : torch.Tensor
        Chemical edges between kept atoms, shape [2, E].
    edge_attr : torch.Tensor
        Edge features of the kept chemical edges, shape [E, edge_in_feats].
    atom2mol : torch.Tensor
        Molecule id of every atom, shape [N], values in [0, M).
    mol2rxn : torch.Tensor
        Reaction id of every molecule, shape [M], values in [0, B).
    mol_side : torch.Tensor
        Side of every molecule, shape [M] (0 = Reactants, 1 = Products).
    mol2side : torch.Tensor
        Side-group id of every molecule, shape [M] (2 * mol2rxn + mol_side).
    side2rxn : torch.Tensor
        Reaction id of every side group, shape [2B] (arange(B) repeated twice).
    side_roles : torch.Tensor
        Role of every side group, shape [2B] (Reactants, Products, ...).
    n_reactions : int
        Number of reactions in the batch, B.
    n_molecules : int
        Number of real molecules in the batch, M.
    """

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    atom2mol: torch.Tensor
    mol2rxn: torch.Tensor
    mol_side: torch.Tensor
    mol2side: torch.Tensor
    side2rxn: torch.Tensor
    side_roles: torch.Tensor
    n_reactions: int
    n_molecules: int


def segment_mean(x: torch.Tensor, index: torch.Tensor, num_segments: int) -> torch.Tensor:
    """
    Average the rows of `x` that belong to each segment.

    Uses native ``index_add_`` on the values and on a count vector, so the
    reduction is a plain segment mean without a scatter/gather dependency.

    Parameters
    ----------
    x : torch.Tensor
        Rows to average, shape [n_rows, D].
    index : torch.Tensor
        Segment id of every row, shape [n_rows], values in [0, num_segments).
    num_segments : int
        Number of segments.

    Returns
    -------
    torch.Tensor
        Segment means, shape [num_segments, D].

    Raises
    ------
    ValueError
        If `index` does not match `x`, or if any segment is empty.
    """
    if index.dim() != 1 or index.shape[0] != x.shape[0]:
        raise ValueError(
            "index must be 1-D with one entry per row of x, got %s and %s"
            % (tuple(index.shape), tuple(x.shape))
        )
    if index.numel() and (int(index.min()) < 0 or int(index.max()) >= num_segments):
        raise ValueError(
            "index values must lie in [0, %d), got range [%d, %d]"
            % (num_segments, int(index.min()), int(index.max()))
        )
    summed = x.new_zeros(num_segments, x.shape[1])
    summed.index_add_(0, index, x)
    counts = x.new_zeros(num_segments)
    counts.index_add_(0, index, torch.ones_like(index, dtype=x.dtype))
    empty = torch.nonzero(counts == 0, as_tuple=False).flatten()
    if empty.numel():
        raise ValueError(
            "empty segment(s) %s: every segment must contain at least one row"
            % empty[:8].tolist()
        )
    return summed / counts.unsqueeze(-1)


def _presence_mask(
    dummy, n_slots: int, n_reactions: int, device, side_name: str
) -> torch.Tensor:
    """
    Turn a per-reaction list of slot flags into a boolean [B, S] mask.

    Parameters
    ----------
    dummy : list
        ``r_dummy`` / ``p_dummy`` as produced by `GraphDataset` and collated by
        `collate_reaction_graphs`: one list of slot flags per reaction.
    n_slots : int
        Number of molecule slots of the side.
    n_reactions : int
        Expected number of reactions.
    device : torch.device
        Device of the returned mask.
    side_name : str
        "reactant" or "product", used in error messages.

    Returns
    -------
    torch.Tensor
        Boolean mask, shape [B, S], True for a real (present) molecule.

    Raises
    ------
    ValueError
        If the mask does not have the expected shape.
    """
    mask = torch.as_tensor(dummy, dtype=torch.bool, device=device)
    if mask.dim() == 1:  # a single reaction given as one flat list of slots
        mask = mask.unsqueeze(0)
    if mask.dim() != 2 or mask.shape[0] != n_reactions or mask.shape[1] != n_slots:
        raise ValueError(
            "%s presence mask has shape %s, expected [%d, %d]"
            % (side_name, tuple(mask.shape), n_reactions, n_slots)
        )
    return mask


def pack_hierarchical_batch(
    rmols: List,
    pmols: List,
    r_dummy: List,
    p_dummy: List,
    device: torch.device,
) -> HierarchicalBatch:
    """
    Pack the per-slot molecule batches into one four-level hierarchy.

    Parameters
    ----------
    rmols : list
        One PyG `Batch` per reactant molecule slot; each holds B molecular
        graphs (padding graphs included).
    pmols : list
        One PyG `Batch` per product molecule slot.
    r_dummy : list
        Per-reaction reactant-slot flags, True where a reactant is present.
    p_dummy : list
        Per-reaction product-slot flags, True where a product is present.
    device : torch.device
        Device for the membership tensors.

    Returns
    -------
    HierarchicalBatch
        The packed real graphs and the atom/molecule/side/reaction membership.

    Raises
    ------
    ValueError
        If the slot batches or the presence masks are inconsistent, if a
        reaction lacks a real molecule on either side, if a real molecule has no
        atom, or if a chemical edge connects two different molecules.
    """
    r_batches = list(rmols)
    p_batches = list(pmols)
    n_r_slots = len(r_batches)
    n_p_slots = len(p_batches)

    if n_r_slots == 0 or n_p_slots == 0:
        raise ValueError(
            "a hierarchical batch needs at least one molecule slot per side, got "
            "%d reactant and %d product slots" % (n_r_slots, n_p_slots)
        )

    r_mask = _presence_mask(r_dummy, n_r_slots, _infer_n_reactions(r_dummy, r_batches),
                            device, "reactant")
    n_reactions = r_mask.shape[0]
    p_mask = _presence_mask(p_dummy, n_p_slots, n_reactions, device, "product")

    _check_slot_batches(r_batches, n_reactions, "reactant")
    _check_slot_batches(p_batches, n_reactions, "product")

    atom_feats = []
    edge_attrs = []
    edge_src = []
    edge_dst = []
    atom2mol_parts = []
    mol2rxn_parts = []
    mol_side_parts = []

    edge_dim = int(r_batches[0].edge_attr.shape[1])
    atom_offset = 0
    mol_offset = 0

    for side, batches, mask in (
        (REACTANTS, r_batches, r_mask),
        (PRODUCTS, p_batches, p_mask),
    ):
        for slot_index, slot in enumerate(batches):
            slot_batch = slot.batch
            keep = mask[:, slot_index][slot_batch]
            if not bool(keep.any()):
                continue

            kept_atoms = torch.nonzero(keep, as_tuple=False).squeeze(1)
            sample_of_atom = slot_batch[kept_atoms]
            # The slot batch concatenates the samples in reaction order and each
            # sample's atoms stay contiguous, so unique_consecutive yields one
            # entry per real molecule occurrence, in reaction order.
            samples, inverse = torch.unique_consecutive(
                sample_of_atom, return_inverse=True
            )
            n_molecules = int(samples.numel())

            local_mol = inverse + mol_offset
            atom2mol_parts.append(local_mol)
            mol2rxn_parts.append(samples)
            mol_side_parts.append(torch.full_like(samples, side))

            atom_feats.append(slot.x[kept_atoms])

            new_index = slot.x.new_full((slot.x.shape[0],), -1, dtype=torch.long)
            new_index[kept_atoms] = (
                torch.arange(kept_atoms.numel(), device=slot.x.device) + atom_offset
            )
            src_new = new_index[slot.edge_index[0]]
            dst_new = new_index[slot.edge_index[1]]
            edge_keep = (src_new >= 0) & (dst_new >= 0)
            src_kept = src_new[edge_keep]
            dst_kept = dst_new[edge_keep]
            if src_kept.numel():
                mol_of_src = local_mol[src_kept - atom_offset]
                mol_of_dst = local_mol[dst_kept - atom_offset]
                if not torch.equal(mol_of_src, mol_of_dst):
                    side_name = "reactant" if side == REACTANTS else "product"
                    raise ValueError(
                        "chemical edge connects two different molecules "
                        "(%s slot %d); edges must stay inside one molecule"
                        % (side_name, slot_index)
                    )
            edge_attrs.append(slot.edge_attr[edge_keep])
            edge_src.append(src_kept)
            edge_dst.append(dst_kept)

            atom_offset += int(kept_atoms.numel())
            mol_offset += n_molecules

    if not atom_feats:
        raise ValueError(
            "the batch contains no real molecule: every slot of every reaction "
            "is padding"
        )

    x = torch.cat(atom_feats, dim=0)
    atom2mol = torch.cat(atom2mol_parts, dim=0)
    mol2rxn = torch.cat(mol2rxn_parts, dim=0)
    mol_side = torch.cat(mol_side_parts, dim=0).to(dtype=torch.long)
    mol2side = 2 * mol2rxn + mol_side
    n_molecules = mol_offset

    if edge_src:
        edge_index = torch.stack([torch.cat(edge_src), torch.cat(edge_dst)], dim=0)
        edge_attr = torch.cat(edge_attrs, dim=0)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long, device=x.device)
        edge_attr = x.new_zeros((0, edge_dim))

    side2rxn = torch.arange(n_reactions, device=x.device).repeat_interleave(2)
    side_roles = torch.arange(2 * n_reactions, device=x.device) % 2

    batch = HierarchicalBatch(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        atom2mol=atom2mol,
        mol2rxn=mol2rxn,
        mol_side=mol_side,
        mol2side=mol2side,
        side2rxn=side2rxn,
        side_roles=side_roles,
        n_reactions=n_reactions,
        n_molecules=n_molecules,
    )
    _validate_packed(batch)
    return batch


def _infer_n_reactions(r_dummy, r_batches) -> int:
    """Reaction count of the batch, read from the slot batches when possible."""
    if len(r_batches):
        return int(r_batches[0].num_graphs)
    mask = torch.as_tensor(r_dummy)
    return int(mask.shape[0]) if mask.dim() else 1


def _check_slot_batches(batches: List, n_reactions: int, side_name: str) -> None:
    """Every slot batch must hold exactly one graph per reaction."""
    for slot_index, slot in enumerate(batches):
        if not hasattr(slot, "batch"):
            raise ValueError(
                "%s slot %d is not a PyG Batch (has no .batch attribute)"
                % (side_name, slot_index)
            )
        if int(slot.num_graphs) != n_reactions:
            raise ValueError(
                "%s slot %d holds %d graphs, expected %d (one per reaction)"
                % (side_name, slot_index, int(slot.num_graphs), n_reactions)
            )


def _validate_packed(batch: HierarchicalBatch) -> None:
    """
    Check the packed hierarchy's membership and edge containment.

    Raises
    ------
    ValueError
        If a length/range invariant is broken, a real molecule has no atom, a
        reaction lacks a real molecule on either side, or a chemical edge
        connects two molecules.
    """
    _validate_membership(batch)
    _validate_segments(batch)
    _validate_edges(batch)


def _validate_membership(batch: HierarchicalBatch) -> None:
    """Check lengths, ranges and the group-node bookkeeping."""
    n_atoms = batch.x.shape[0]
    n_molecules = batch.n_molecules
    n_reactions = batch.n_reactions

    if batch.atom2mol.shape[0] != n_atoms:
        raise ValueError("atom2mol length does not match the atom count")
    if n_atoms and (
        int(batch.atom2mol.min()) < 0 or int(batch.atom2mol.max()) >= n_molecules
    ):
        raise ValueError("atom2mol values are outside [0, %d)" % n_molecules)

    for name, tensor in (
        ("mol2rxn", batch.mol2rxn),
        ("mol_side", batch.mol_side),
        ("mol2side", batch.mol2side),
    ):
        if tensor.shape[0] != n_molecules:
            raise ValueError("%s length does not match the molecule count" % name)
    if n_molecules and (
        int(batch.mol2rxn.min()) < 0 or int(batch.mol2rxn.max()) >= n_reactions
    ):
        raise ValueError("mol2rxn values are outside [0, %d)" % n_reactions)
    if n_molecules and not torch.isin(
        batch.mol_side,
        torch.tensor([REACTANTS, PRODUCTS], device=batch.mol_side.device),
    ).all():
        raise ValueError("mol_side values must be 0 (Reactants) or 1 (Products)")
    if not torch.equal(batch.mol2side, 2 * batch.mol2rxn + batch.mol_side):
        raise ValueError("mol2side is not 2 * mol2rxn + mol_side")

    expected_side2rxn = torch.arange(
        n_reactions, device=batch.side2rxn.device
    ).repeat_interleave(2)
    if batch.side2rxn.shape[0] != 2 * n_reactions or not torch.equal(
        batch.side2rxn, expected_side2rxn
    ):
        raise ValueError(
            "side2rxn must be arange(B).repeat_interleave(2) "
            "(Reactants before Products for every reaction)"
        )


def _validate_segments(batch: HierarchicalBatch) -> None:
    """Check that every molecule has atoms and every reaction has both sides."""
    atom_counts = x_counts(batch.atom2mol, batch.n_molecules, batch.x.dtype)
    if batch.n_molecules and bool((atom_counts == 0).any()):
        raise ValueError(
            "molecule(s) %s have no atom"
            % torch.nonzero(atom_counts == 0, as_tuple=False).flatten()[:8].tolist()
        )

    present = torch.zeros(
        (batch.n_reactions, 2), dtype=torch.bool, device=batch.mol_side.device
    )
    if batch.n_molecules:
        present[batch.mol2rxn, batch.mol_side] = True
    missing = torch.nonzero(~present)
    if missing.numel():
        reaction, side = missing[0].tolist()
        raise ValueError(
            "reaction %d has no real molecule on the %s side"
            % (reaction, "Reactants" if side == REACTANTS else "Products")
        )


def _validate_edges(batch: HierarchicalBatch) -> None:
    """Check that no chemical edge connects two different molecules."""
    if not batch.edge_index.numel():
        return
    src_mol = batch.atom2mol[batch.edge_index[0]]
    dst_mol = batch.atom2mol[batch.edge_index[1]]
    if not torch.equal(src_mol, dst_mol):
        raise ValueError("a chemical edge connects two different molecules")


def x_counts(index: torch.Tensor, num_segments: int, dtype) -> torch.Tensor:
    """Number of rows per segment (helper used by the validation above)."""
    counts = torch.zeros(num_segments, dtype=dtype, device=index.device)
    if index.numel():
        counts.index_add_(0, index, torch.ones_like(index, dtype=dtype))
    return counts
