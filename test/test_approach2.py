"""
CPU-light tests for Approach 2: pairwise reactant tokens and selectable regression heads.
"""

import copy
import logging
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from attention import ReactionSelfAttention  # noqa: E402
from data import GraphDataset  # noqa: E402
from finetune import _build_model, _load_checkpoint_safely  # noqa: E402
from gin import GIN  # noqa: E402
from main_finetune import build_parser  # noqa: E402
from model import (  # noqa: E402
    ATTENTION_TARGETS,
    HEADS,
    REACTANT_TOKENS,
    REACTION_COMBINE_DIMS,
    model,
)
from reaction_data import get_graph_data  # noqa: E402
from utils import collate_reaction_graphs, set_seed  # noqa: E402


# ----------------------------------------------------------------------------
# 1. Tests for _add_pair_tokens
# ----------------------------------------------------------------------------


def test_add_pair_tokens_shape():
    """_add_pair_tokens produces the exact expected output shapes."""
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 4, 8
    expected_expanded_S = S + S * (S - 1) // 2  # 4 + 6 = 10

    tokens = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)
    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    assert exp_tokens.shape == (B, expected_expanded_S, D)
    assert exp_mask.shape == (B, expected_expanded_S)
    assert len(token_slots) == expected_expanded_S


def test_add_pair_tokens_ordering_and_arithmetic():
    """Pair tokens follow triu_indices order and equal r_i + r_j."""
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 4, 8
    tokens = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)

    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    # First S entries are individual slots (0,), (1,), ..., (S-1,)
    for s in range(S):
        assert token_slots[s] == (s,)
        torch.testing.assert_close(exp_tokens[:, s], tokens[:, s])

    # Subsequent entries are unordered pairs following torch.triu_indices
    i_indices, j_indices = torch.triu_indices(S, S, offset=1)
    pair_idx = S
    for u, v in zip(i_indices.tolist(), j_indices.tolist()):
        assert token_slots[pair_idx] == (u, v)
        expected_pair = tokens[:, u] + tokens[:, v]
        torch.testing.assert_close(exp_tokens[:, pair_idx], expected_pair)
        pair_idx += 1


def test_add_pair_tokens_mask_with_padding():
    """Every pair slot touching padding is masked False."""
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 4, 8
    tokens = torch.randn(B, S, D)
    # Batch 0 has 2 real reactants (slots 0, 1) and 2 padded (slots 2, 3)
    # Batch 1 has 3 real reactants (slots 0, 1, 2) and 1 padded (slot 3)
    mask = torch.tensor(
        [
            [True, True, False, False],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )

    _, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    for b in range(B):
        for k, slot in enumerate(token_slots):
            if len(slot) == 1:
                assert exp_mask[b, k] == mask[b, slot[0]]
            else:
                u, v = slot
                expected = mask[b, u] and mask[b, v]
                assert exp_mask[b, k] == expected


def test_add_pair_tokens_single_slot():
    """S=1 produces no pair tokens and retains shape [B, 1, D]."""
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 1, 8
    tokens = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)

    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)
    assert exp_tokens.shape == (B, 1, D)
    assert exp_mask.shape == (B, 1)
    assert token_slots == [(0,)]
    torch.testing.assert_close(exp_tokens, tokens)


def test_add_pair_tokens_single_real_reactant_with_padding():
    """Multiple slots with only one real reactant mask every pair."""
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 1, 3, 8
    tokens = torch.randn(B, S, D)
    mask = torch.tensor([[True, False, False]], dtype=torch.bool)

    _, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)
    # Pairs are (0,1), (0,2), (1,2)
    # Real individual is slot 0; all pairs touch padding so must be False
    assert exp_mask[0, 0].item() is True
    assert exp_mask[0, 1].item() is False
    assert exp_mask[0, 2].item() is False
    assert (exp_mask[0, 3:] == False).all()  # all pair slots False


# ----------------------------------------------------------------------------
# 2. Head tests
# ----------------------------------------------------------------------------


def test_linear_head():
    """head='linear' creates exactly nn.Linear with standard state_dict keys."""
    net = model(10, 5, num_layer=1, emb_dim=16, drop_ratio=0.0, head="linear")
    assert isinstance(net.regressor, nn.Linear)
    assert net.regressor.in_features == 16 * 2  # concat default
    assert net.regressor.out_features == 1

    keys = list(net.regressor.state_dict().keys())
    assert keys == ["weight", "bias"]


def test_mlp_head():
    """head='mlp' creates the exact requested nn.Sequential layers."""
    emb_dim = 16
    drop_ratio = 0.1
    net = model(
        10,
        5,
        num_layer=1,
        emb_dim=emb_dim,
        drop_ratio=drop_ratio,
        reaction_combine="concat",
        head="mlp",
    )
    assert isinstance(net.regressor, nn.Sequential)
    assert len(net.regressor) == 4

    l1, act, drop, l2 = net.regressor
    assert isinstance(l1, nn.Linear)
    assert l1.in_features == 2 * emb_dim
    assert l1.out_features == emb_dim

    assert isinstance(act, nn.ReLU)

    assert isinstance(drop, nn.Dropout)
    assert drop.p == drop_ratio

    assert isinstance(l2, nn.Linear)
    assert l2.in_features == emb_dim
    assert l2.out_features == 1

    # Forward output shape is [B]
    B = 2
    dummy_input = torch.randn(B, 2 * emb_dim)
    out = net.regressor(dummy_input).squeeze(-1)
    assert out.shape == (B,)


# ----------------------------------------------------------------------------
# 3. Full-model test with real featurization
# ----------------------------------------------------------------------------


def _build_real_dataset_and_batch(reactions, rmol_max_cnt=2, pmol_max_cnt=1):
    """Helper featurizing real SMILES reactions with get_graph_data."""
    rmol, pmol, rxn = get_graph_data(
        reactions,
        rmol_max_cnt=rmol_max_cnt,
        pmol_max_cnt=pmol_max_cnt,
        y_list=[1.0] * len(reactions),
    )
    ds = GraphDataset(rmol=rmol, pmol=pmol, reaction=rxn)
    batch = collate_reaction_graphs([ds[i] for i in range(len(reactions))])
    rmols = list(batch[:rmol_max_cnt])
    pmols = list(batch[rmol_max_cnt : rmol_max_cnt + pmol_max_cnt])
    r_dummy = batch[-4]
    p_dummy = batch[-3]
    labels = batch[-2]
    node_dim = ds.rmol_node_attr[0].shape[1]
    edge_dim = ds.rmol_edge_attr[0].shape[1]
    return ds, rmols, pmols, r_dummy, p_dummy, labels, node_dim, edge_dim


def test_full_model_invariances_and_gradient():
    """Verify permutation, padding, noise invariance and gradient flow."""
    set_seed(42)
    device = torch.device("cpu")

    # 2 tiny valid reactions:
    # Rxn 1: ethanol + acetic acid -> ethyl acetate
    # Rxn 2: ethylamine + acetyl chloride -> N-ethylacetamide
    reactions = [
        "CCO.CC(=O)O>>CCOC(=O)C",
        "CCN.CC(=O)Cl>>CCNC(=O)C",
    ]

    _, rmols, pmols, r_dummy, p_dummy, _, node_dim, edge_dim = (
        _build_real_dataset_and_batch(reactions, rmol_max_cnt=2, pmol_max_cnt=1)
    )

    net = model(
        node_in_feats=node_dim,
        edge_in_feats=edge_dim,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        reactant_tokens="comb",
        head="mlp",
        attention_on="reactants",
    ).to(device)
    net.eval()

    # Base prediction
    with torch.no_grad():
        base_pred, base_rvecs = net(rmols, pmols, r_dummy, p_dummy, device)

    # 1. Permutation invariance: swap reactant slot 0 and 1
    rmols_perm = [rmols[1], rmols[0]]
    r_dummy_perm = [[d[1], d[0]] for d in r_dummy]
    with torch.no_grad():
        perm_pred, perm_rvecs = net(rmols_perm, pmols, r_dummy_perm, p_dummy, device)

    torch.testing.assert_close(
        perm_pred, base_pred, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.tensor(perm_rvecs), torch.tensor(base_rvecs), rtol=1e-5, atol=1e-5
    )

    # 2. Padding slot invariance: featurize with 3 reactant slots (slot 2 is dummy)
    _, rmols_pad, pmols_pad, r_dummy_pad, p_dummy_pad, _, _, _ = (
        _build_real_dataset_and_batch(reactions, rmol_max_cnt=3, pmol_max_cnt=1)
    )
    with torch.no_grad():
        pad_pred, pad_rvecs = net(rmols_pad, pmols_pad, r_dummy_pad, p_dummy_pad, device)

    torch.testing.assert_close(
        pad_pred, base_pred, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.tensor(pad_rvecs), torch.tensor(base_rvecs), rtol=1e-5, atol=1e-5
    )

    # 3. Noise invariance on padded graph: add arbitrary non-zero noise to rmols_pad[2]
    noisy_rmols_pad = copy.deepcopy(rmols_pad)
    noisy_rmols_pad[2].x = noisy_rmols_pad[2].x + 42.0
    noisy_rmols_pad[2].edge_attr = noisy_rmols_pad[2].edge_attr + 17.0
    with torch.no_grad():
        noisy_pred, noisy_rvecs = net(
            noisy_rmols_pad, pmols_pad, r_dummy_pad, p_dummy_pad, device
        )

    torch.testing.assert_close(
        noisy_pred, base_pred, rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(
        torch.tensor(noisy_rvecs), torch.tensor(base_rvecs), rtol=1e-5, atol=1e-5
    )

    # 4. Gradient backpropagation
    net.train()
    net.zero_grad()
    pred, _ = net(rmols, pmols, r_dummy, p_dummy, device)
    loss = pred.sum()
    loss.backward()

    # At least one GIN parameter must receive a finite, non-zero gradient
    gin_grads = [p.grad for p in net.gnn.parameters() if p.grad is not None]
    assert len(gin_grads) > 0
    assert any((g != 0).any() and torch.isfinite(g).all() for g in gin_grads)


# ----------------------------------------------------------------------------
# 4. Backward-compatibility tests
# ----------------------------------------------------------------------------


class OriginalModelBaseline(nn.Module):
    """
    Compact frozen copy of the original model (commit 3af22da) before Approach 2.
    """

    def __init__(
        self,
        node_in_feats: int,
        edge_in_feats: int,
        num_layer: int,
        emb_dim: int,
        drop_ratio: float,
        num_attention_layer: int = 1,
        num_heads: int = 1,
        reaction_combine: str = "concat",
        attention_on: str = "reactants",
    ) -> None:
        super().__init__()
        self.config = {
            "node_in_feats": int(node_in_feats),
            "edge_in_feats": int(edge_in_feats),
            "num_layer": int(num_layer),
            "emb_dim": int(emb_dim),
            "drop_ratio": float(drop_ratio),
            "num_attention_layer": int(num_attention_layer),
            "num_heads": int(num_heads),
            "reaction_combine": str(reaction_combine),
            "attention_on": str(attention_on),
        }
        self.gnn = GIN(node_in_feats, edge_in_feats, num_layer, emb_dim, drop_ratio)
        self.attention_on = attention_on
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )
        self.reaction_combine = reaction_combine
        self.regressor = nn.Linear(REACTION_COMBINE_DIMS[reaction_combine] * emb_dim, 1)
        self.store_attention = False
        self.last_attention = None

    def _attend(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        for attention_layer in self.attention_layers[:-1]:
            tokens = tokens + attention_layer(tokens, mask=mask)
        attended = self.attention_layers[-1](
            tokens, mask=mask, return_weights=self.store_attention
        )
        if self.store_attention:
            attended, att_weights = attended
            return attended, att_weights.detach().cpu().tolist()
        return attended, None

    @staticmethod
    def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.unsqueeze(-1).to(x.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / count

    def _combine(self, r: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.cat((r, p), dim=1)

    def forward(self, rmols, pmols, r_dummy, p_dummy, device):
        r_tokens = torch.stack([self.gnn(rmol) for rmol in rmols], dim=1).to(device)
        p_tokens = torch.stack([self.gnn(pmol) for pmol in pmols], dim=1).to(device)
        r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
        p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)
        weights = {}
        if self.attention_on == "reactants":
            r_tokens, weights["reactants"] = self._attend(r_tokens, r_mask)
        reactant_vectors = self._masked_mean(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)
        reaction_vectors = self._combine(reactant_vectors, product_vectors)
        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()


def test_reproduces_original_model_exactly():
    """Default new model with identical weights produces same outputs as original."""
    set_seed(123)
    reactions = ["CCO.CC(=O)O>>CCOC(=O)C", "CCN.CC(=O)Cl>>CCNC(=O)C"]
    _, rmols, pmols, r_dummy, p_dummy, _, node_dim, edge_dim = (
        _build_real_dataset_and_batch(reactions, 2, 1)
    )
    device = torch.device("cpu")

    orig = OriginalModelBaseline(node_dim, edge_dim, 1, 16, 0.0).to(device)
    new_m = model(node_dim, edge_dim, 1, 16, 0.0).to(device)

    # Load original weights into new model
    new_m.load_state_dict(orig.state_dict(), strict=True)

    orig.eval()
    new_m.eval()
    with torch.no_grad():
        out_orig, rvecs_orig = orig(rmols, pmols, r_dummy, p_dummy, device)
        out_new, rvecs_new = new_m(rmols, pmols, r_dummy, p_dummy, device)

    torch.testing.assert_close(out_new, out_orig)
    torch.testing.assert_close(torch.tensor(rvecs_new), torch.tensor(rvecs_orig))


def test_from_config_backward_compatibility():
    """Old config without reactant_tokens or head resolves to ind/linear."""
    old_config = {
        "node_in_feats": 155,
        "edge_in_feats": 9,
        "num_layer": 2,
        "emb_dim": 32,
        "drop_ratio": 0.0,
        "num_attention_layer": 1,
        "num_heads": 1,
        "reaction_combine": "concat",
        "attention_on": "reactants",
    }
    net = model.from_config(old_config)
    assert net.reactant_tokens == "ind"
    assert net.head == "linear"
    assert isinstance(net.regressor, nn.Linear)
    assert net.config["reactant_tokens"] == "ind"
    assert net.config["head"] == "linear"


def test_cli_parsing():
    """CLI defaults and explicit choices parse properly."""
    parser = build_parser()
    args_default = parser.parse_args([])
    assert args_default.reactant_tokens == "ind"
    assert args_default.head == "linear"

    args_custom = parser.parse_args(["--reactant_tokens", "comb", "--head", "mlp"])
    assert args_custom.reactant_tokens == "comb"
    assert args_custom.head == "mlp"


def test_load_checkpoint_safely_compatibility():
    """_load_checkpoint_safely handles old checkpoints, warm-starts, and errors."""
    logger = logging.getLogger("test_logger")

    # 1. Old checkpoint without reactant_tokens/head loads into default model
    base_m = model(10, 5, 1, 16, 0.0, head="linear")
    old_ckpt = {
        "model_state_dict": base_m.state_dict(),
        "model_config": {
            "node_in_feats": 10,
            "edge_in_feats": 5,
            "num_layer": 1,
            "emb_dim": 16,
            "drop_ratio": 0.0,
            "num_attention_layer": 1,
            "num_heads": 1,
            "reaction_combine": "concat",
            "attention_on": "reactants",
        },
    }
    target_net = model(10, 5, 1, 16, 0.0, head="linear")
    _load_checkpoint_safely(target_net, old_ckpt, logger)

    # 2. reactant_tokens mismatch must raise RuntimeError
    comb_net = model(10, 5, 1, 16, 0.0, reactant_tokens="comb")
    with pytest.raises(RuntimeError, match="different architecture"):
        _load_checkpoint_safely(comb_net, old_ckpt, logger)

    # 3. head mismatch (linear checkpoint -> mlp model) allows warm-start
    mlp_net = model(10, 5, 1, 16, 0.0, head="mlp")
    _load_checkpoint_safely(mlp_net, old_ckpt, logger)

    # 4. Encoder mismatch must raise RuntimeError
    wrong_encoder_ckpt = copy.deepcopy(old_ckpt)
    del wrong_encoder_ckpt["model_config"]  # bypass config check
    state_dict = copy.deepcopy(base_m.state_dict())
    state_dict["gnn.project_node_feats.0.weight"] = torch.randn(16, 999)
    wrong_encoder_ckpt["model_state_dict"] = state_dict
    with pytest.raises(RuntimeError):
        _load_checkpoint_safely(target_net, wrong_encoder_ckpt, logger)
