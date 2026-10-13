"""
Hierarchical reaction graph models for reaction-yield prediction.

Two architectures are provided; both share the explicit batch packing in
`hierarchical_data` and the GINE helpers in `gin`:

    Hierarchical4YieldModel ("hierarchical4")
        atoms -> molecules -> side groups -> reaction, predicting from the
        reaction node.

    Hierarchical3YieldModel ("hierarchical3")
        atoms -> molecules -> side groups (Reactants / Products), with no
        reaction node: the yield is predicted from the concatenation of the two
        final group embeddings [Reactants, Products].

The supplied left-side molecules all belong to Reactants (catalysts, ligands,
solvents, bases, reagents and additives included); the supplied right-side
molecules all belong to Products. Grouping depends only on the reaction side.

Both models are iterative: each pass runs the upward steps, and, between passes
only, the downward feedback steps, so the next pass starts from feedback-updated
atom states. Neither model uses attention, pair tokens, relation-network
pooling, a sigmoid output or target normalization.
"""

from typing import List

import torch
import torch.nn as nn

from gin import build_gine_conv_stack, propagate_gine
from hierarchical_data import pack_hierarchical_batch, segment_mean
from model import HEADS


def _cross_level_mlp(in_dim: int, emb_dim: int, dropout: float) -> nn.Sequential:
    """
    Concatenation -> emb_dim -> emb_dim update block.

    Parameters
    ----------
    in_dim : int
        Width of the concatenated input (2D, or 3D for groups-to-reaction).
    emb_dim : int
        Hidden/embedding dimension D.
    dropout : float
        Dropout probability.

    Returns
    -------
    nn.Sequential
        Linear -> ReLU -> Dropout -> Linear.
    """
    return nn.Sequential(
        nn.Linear(in_dim, emb_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(emb_dim, emb_dim),
    )


class Hierarchical4YieldModel(nn.Module):
    """
    Atom/molecule/side/reaction hierarchy for reaction-yield regression.

    Parameters
    ----------
    node_in_feats : int
        Input feature dimension of the atoms.
    edge_in_feats : int
        Input feature dimension of the chemical bonds.
    emb_dim : int
        Embedding dimension D of every level.
    drop_ratio : float
        Dropout probability.
    hierarchy_passes : int, optional
        Number K of upward passes (default is 2). K=1 is the collect-only
        control without any feedback module.
    hierarchy_gine_layers : int, optional
        Number of GINE layers per pass (default is 2).
    hierarchy_no_feedback : bool, optional
        Skip the downward feedback updates while keeping K upward passes
        (default is False, i.e. feedback between consecutive passes).
    """

    YIELD_ARCH = "hierarchical4"

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        emb_dim: int,
        drop_ratio: float,
        hierarchy_passes: int = 2,
        hierarchy_gine_layers: int = 2,
        hierarchy_no_feedback: bool = False,
    ) -> None:
        super(Hierarchical4YieldModel, self).__init__()

        if int(hierarchy_passes) < 1:
            raise ValueError(
                "hierarchy_passes must be a positive integer, got %d"
                % hierarchy_passes
            )
        if int(hierarchy_gine_layers) < 1:
            raise ValueError(
                "hierarchy_gine_layers must be a positive integer, got %d"
                % hierarchy_gine_layers
            )

        self.yield_arch = self.YIELD_ARCH
        self.emb_dim = int(emb_dim)
        self.feedback_enabled = not bool(hierarchy_no_feedback)
        self.hierarchy_passes = int(hierarchy_passes)
        self.hierarchy_gine_layers = int(hierarchy_gine_layers)

        # Everything needed to rebuild this architecture; saved in checkpoints.
        self.config = {
            "yield_arch": self.YIELD_ARCH,
            "node_in_feats": int(node_in_feats),
            "edge_in_feats": int(edge_in_feats),
            "emb_dim": int(emb_dim),
            "drop_ratio": float(drop_ratio),
            "hierarchy_passes": int(hierarchy_passes),
            "hierarchy_gine_layers": int(hierarchy_gine_layers),
            "hierarchy_no_feedback": bool(hierarchy_no_feedback),
        }

        D = self.emb_dim
        K = self.hierarchy_passes

        # One atom encoder and one bond encoder, applied once before the passes.
        self.atom_encoder = nn.Sequential(nn.Linear(node_in_feats, D), nn.ReLU())
        self.bond_encoder = nn.Linear(edge_in_feats, D)

        # Learned initial states of the three coarser levels.
        self.molecule_role_embedding = nn.Embedding(2, D)
        self.side_role_embedding = nn.Embedding(2, D)
        self.reaction_token = nn.Parameter(torch.zeros(D))

        # K independent GINE stacks, each consuming the current atom states.
        self.gine_stacks = nn.ModuleList(
            [
                build_gine_conv_stack(self.hierarchy_gine_layers, D)
                for _ in range(K)
            ]
        )
        self.dropout = nn.Dropout(drop_ratio)

        # K independent upward update sets.
        self.atom_to_molecule = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(K)]
        )
        self.ln_m_up = nn.ModuleList([nn.LayerNorm(D) for _ in range(K)])
        self.molecule_to_side = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(K)]
        )
        self.ln_s_up = nn.ModuleList([nn.LayerNorm(D) for _ in range(K)])
        self.side_to_reaction = nn.ModuleList(
            [_cross_level_mlp(3 * D, D, drop_ratio) for _ in range(K)]
        )
        self.ln_r_up = nn.ModuleList([nn.LayerNorm(D) for _ in range(K)])

        # Yield head: Linear(D, D) -> ReLU -> Dropout -> Linear(D, 1).
        self.regressor = nn.Sequential(
            nn.Linear(D, D),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(D, 1),
        )

        # Exactly K - 1 downward feedback sets, created last so that the
        # feedback-disabled control shares their weights with an identical seed.
        n_down = (K - 1) if self.feedback_enabled else 0
        self.reaction_to_side = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(n_down)]
        )
        self.ln_s_down = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_down)])
        self.side_to_molecule = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(n_down)]
        )
        self.ln_m_down = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_down)])
        self.molecule_to_atom = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(n_down)]
        )
        self.ln_h_down = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_down)])

    @classmethod
    def from_config(cls, config: dict) -> "Hierarchical4YieldModel":
        """
        Rebuild a model from the `config` of another one.

        Parameters
        ----------
        config : dict
            Constructor arguments, as stored in `model.config`.

        Returns
        -------
        Hierarchical4YieldModel
            A freshly initialised model with that architecture.
        """
        config = dict(config)
        config.pop("yield_arch", None)
        return cls(**config)

    def forward(
        self,
        rmols: List,
        pmols: List,
        r_dummy: List,
        p_dummy: List,
        device: torch.device,
    ) -> tuple:
        """
        Forward pass of the hierarchical model.

        Parameters
        ----------
        rmols : list
            Reactant molecule slot batches.
        pmols : list
            Product molecule slot batches.
        r_dummy : list
            Per-reaction reactant-slot flags (True = real molecule).
        p_dummy : list
            Per-reaction product-slot flags (True = real molecule).
        device : torch.device
            Device to run the computations.

        Returns
        -------
        tuple
            Predicted yields of shape [batch_size] and the reaction states as a
            Python list of shape [batch_size, emb_dim].
        """
        batch = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, device)
        D = self.emb_dim
        B = batch.n_reactions
        M = batch.n_molecules

        h = self.atom_encoder(batch.x)
        e = self.bond_encoder(batch.edge_attr)

        m = self.molecule_role_embedding(batch.mol_side)
        s = self.side_role_embedding(batch.side_roles)
        r = self.reaction_token.expand(B, D)

        for k in range(self.hierarchy_passes):
            # 1. Chemical-bond propagation on the current atom states.
            h = propagate_gine(
                self.gine_stacks[k], h, batch.edge_index, e, self.dropout
            )

            # 2. Atoms -> molecules.
            a = segment_mean(h, batch.atom2mol, M)
            m = self.ln_m_up[k](
                m + self.atom_to_molecule[k](torch.cat((m, a), dim=-1))
            )

            # 3. Molecules -> their own side groups.
            u = segment_mean(m, batch.mol2side, 2 * B)
            s = self.ln_s_up[k](
                s + self.molecule_to_side[k](torch.cat((s, u), dim=-1))
            )

            # 4. Ordered Reactants/Products -> reaction.
            pairs = s.reshape(B, 2, D)
            r = self.ln_r_up[k](
                r
                + self.side_to_reaction[k](
                    torch.cat((r, pairs[:, 0], pairs[:, 1]), dim=-1)
                )
            )

            if self.feedback_enabled and k < self.hierarchy_passes - 1:
                # 5. Reaction -> both side groups.
                s = self.ln_s_down[k](
                    s
                    + self.reaction_to_side[k](
                        torch.cat((s, r[batch.side2rxn]), dim=-1)
                    )
                )

                # 6. Each side -> its own molecules.
                m = self.ln_m_down[k](
                    m
                    + self.side_to_molecule[k](
                        torch.cat((m, s[batch.mol2side]), dim=-1)
                    )
                )

                # 7. Each molecule -> its own atoms.
                h = self.ln_h_down[k](
                    h
                    + self.molecule_to_atom[k](
                        torch.cat((h, m[batch.atom2mol]), dim=-1)
                    )
                )

        pred = self.regressor(r).squeeze(-1)
        return pred, r.detach().cpu().tolist()


class Hierarchical3YieldModel(nn.Module):
    """
    Three-level atom/molecule/side-group hierarchy for reaction-yield regression.

    The hierarchy is

        Level 1  atoms connected by chemical bonds inside one molecule
        Level 2  one node per real supplied molecular graph
        Level 3  exactly two group nodes per reaction: Reactants and Products

    There is no reaction node. Each pass runs the three upward steps

        bonds -> molecules -> side groups

    and, between passes only, the two downward feedback steps

        side groups -> molecules -> atoms

    so the next pass starts from feedback-updated atom states. The final pass
    stops after the group update; the two group embeddings are concatenated as
    [Reactants, Products] and fed to the prediction head. Reactants and Products
    only ever interact through that head (there is no cross-side message
    passing).

    Parameters
    ----------
    node_in_feats : int
        Input feature dimension of the atoms.
    edge_in_feats : int
        Input feature dimension of the chemical bonds.
    emb_dim : int
        Embedding dimension D of every level.
    drop_ratio : float
        Dropout probability.
    hierarchy_passes : int, optional
        Number K of upward passes (default is 2). K=1 is the collect-only
        control without any feedback module.
    hierarchy_gine_layers : int, optional
        Number of GINE layers per pass (default is 2).
    hierarchy_no_feedback : bool, optional
        Skip the side-group -> molecule -> atom feedback updates while keeping K
        upward passes (default is False, i.e. feedback between consecutive
        passes).
    head : str, optional
        Regression head (default is "linear"): "linear" for Linear(2D, 1), "mlp"
        for Linear(2D, D) -> ReLU -> Dropout -> Linear(D, 1).
    """

    YIELD_ARCH = "hierarchical3"

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        emb_dim: int,
        drop_ratio: float,
        hierarchy_passes: int = 2,
        hierarchy_gine_layers: int = 2,
        hierarchy_no_feedback: bool = False,
        head: str = "linear",
    ) -> None:
        super(Hierarchical3YieldModel, self).__init__()

        if int(hierarchy_passes) < 1:
            raise ValueError(
                "hierarchy_passes must be a positive integer, got %d"
                % hierarchy_passes
            )
        if int(hierarchy_gine_layers) < 1:
            raise ValueError(
                "hierarchy_gine_layers must be a positive integer, got %d"
                % hierarchy_gine_layers
            )
        if head not in HEADS:
            raise ValueError(
                "head must be one of %s, got %r" % (list(HEADS), head)
            )

        self.yield_arch = self.YIELD_ARCH
        self.emb_dim = int(emb_dim)
        self.feedback_enabled = not bool(hierarchy_no_feedback)
        self.hierarchy_passes = int(hierarchy_passes)
        self.hierarchy_gine_layers = int(hierarchy_gine_layers)
        self.head = str(head)

        # Everything needed to rebuild this architecture; saved in checkpoints.
        self.config = {
            "yield_arch": self.YIELD_ARCH,
            "node_in_feats": int(node_in_feats),
            "edge_in_feats": int(edge_in_feats),
            "emb_dim": int(emb_dim),
            "drop_ratio": float(drop_ratio),
            "hierarchy_passes": int(hierarchy_passes),
            "hierarchy_gine_layers": int(hierarchy_gine_layers),
            "hierarchy_no_feedback": bool(hierarchy_no_feedback),
            "head": str(head),
        }

        D = self.emb_dim
        K = self.hierarchy_passes

        # One atom encoder and one bond encoder, applied once before the passes.
        self.atom_encoder = nn.Sequential(nn.Linear(node_in_feats, D), nn.ReLU())
        self.bond_encoder = nn.Linear(edge_in_feats, D)

        # Learned initial states of the two coarser levels.
        self.molecule_role_embedding = nn.Embedding(2, D)
        self.side_role_embedding = nn.Embedding(2, D)

        # K independent GINE stacks, each consuming the current atom states.
        self.gine_stacks = nn.ModuleList(
            [
                build_gine_conv_stack(self.hierarchy_gine_layers, D)
                for _ in range(K)
            ]
        )
        self.dropout = nn.Dropout(drop_ratio)

        # K independent upward update sets.
        self.atom_to_molecule = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(K)]
        )
        self.ln_m_up = nn.ModuleList([nn.LayerNorm(D) for _ in range(K)])
        self.molecule_to_side = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(K)]
        )
        self.ln_s_up = nn.ModuleList([nn.LayerNorm(D) for _ in range(K)])

        # Exactly K - 1 feedback sets group -> molecule -> atom. Created before
        # the head so that the whole encoder/backbone is identical between two
        # runs that differ only in the prediction head.
        n_down = (K - 1) if self.feedback_enabled else 0
        self.side_to_molecule = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(n_down)]
        )
        self.ln_m_down = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_down)])
        self.molecule_to_atom = nn.ModuleList(
            [_cross_level_mlp(2 * D, D, drop_ratio) for _ in range(n_down)]
        )
        self.ln_h_down = nn.ModuleList([nn.LayerNorm(D) for _ in range(n_down)])

        # Yield head, created last so that it is the only difference between the
        # linear and the MLP run. Only the selected head is allocated.
        if head == "linear":
            self.regressor = nn.Linear(2 * D, 1)
        else:
            self.regressor = nn.Sequential(
                nn.Linear(2 * D, D),
                nn.ReLU(),
                nn.Dropout(drop_ratio),
                nn.Linear(D, 1),
            )

    @classmethod
    def from_config(cls, config: dict) -> "Hierarchical3YieldModel":
        """
        Rebuild a model from the `config` of another one.

        Parameters
        ----------
        config : dict
            Constructor arguments, as stored in `model.config`.

        Returns
        -------
        Hierarchical3YieldModel
            A freshly initialised model with that architecture.
        """
        config = dict(config)
        config.pop("yield_arch", None)
        return cls(**config)

    def forward(
        self,
        rmols: List,
        pmols: List,
        r_dummy: List,
        p_dummy: List,
        device: torch.device,
    ) -> tuple:
        """
        Forward pass of the three-level hierarchical model.

        Parameters
        ----------
        rmols : list
            Reactant molecule slot batches.
        pmols : list
            Product molecule slot batches.
        r_dummy : list
            Per-reaction reactant-slot flags (True = real molecule).
        p_dummy : list
            Per-reaction product-slot flags (True = real molecule).
        device : torch.device
            Device to run the computations.

        Returns
        -------
        tuple
            Predicted yields of shape [batch_size] and the concatenated
            [Reactants, Products] readout of shape [batch_size, 2 * emb_dim] as
            a Python list.
        """
        batch = pack_hierarchical_batch(rmols, pmols, r_dummy, p_dummy, device)
        D = self.emb_dim
        B = batch.n_reactions
        M = batch.n_molecules

        # Encode the raw atom features once; later passes consume the current
        # (feedback-updated) atom states h.
        h = self.atom_encoder(batch.x)
        e = self.bond_encoder(batch.edge_attr)

        m = self.molecule_role_embedding(batch.mol_side)
        s = self.side_role_embedding(batch.side_roles)

        for k in range(self.hierarchy_passes):
            # 1. Chemical-bond propagation on the current atom states.
            h = propagate_gine(
                self.gine_stacks[k], h, batch.edge_index, e, self.dropout
            )

            # 2. Atoms -> molecules.
            a = segment_mean(h, batch.atom2mol, M)
            m = self.ln_m_up[k](
                m + self.atom_to_molecule[k](torch.cat((m, a), dim=-1))
            )

            # 3. Molecules -> their own side groups.
            u = segment_mean(m, batch.mol2side, 2 * B)
            s = self.ln_s_up[k](
                s + self.molecule_to_side[k](torch.cat((s, u), dim=-1))
            )

            if self.feedback_enabled and k < self.hierarchy_passes - 1:
                # 4. Each side -> its own molecules.
                m = self.ln_m_down[k](
                    m
                    + self.side_to_molecule[k](
                        torch.cat((m, s[batch.mol2side]), dim=-1)
                    )
                )

                # 5. Each molecule -> its own atoms.
                h = self.ln_h_down[k](
                    h
                    + self.molecule_to_atom[k](
                        torch.cat((h, m[batch.atom2mol]), dim=-1)
                    )
                )

        # Concatenated readout: Reactants first, then Products.
        side_pairs = s.reshape(B, 2, D)
        prediction_vector = torch.cat(
            (side_pairs[:, 0], side_pairs[:, 1]), dim=-1
        )
        pred = self.regressor(prediction_vector).squeeze(-1)
        return pred, prediction_vector.detach().cpu().tolist()
