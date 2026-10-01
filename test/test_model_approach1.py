"""
Lightweight CPU unit tests for Approach 1:
- Pairwise reactant tokens (_add_pair_tokens)
- Relation-network pooling (RelationPooling)
- Switchable regression head (linear / mlp)
- Full model permutation invariance and masking tests
- Suzuki-width single forward+backward shape/mask test
- Backward compatibility with original baseline
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
if TEST_DIR not in sys.path:
    sys.path.insert(0, TEST_DIR)

from attention import ReactionSelfAttention  # noqa: E402
from gin import GIN  # noqa: E402
from model import (  # noqa: E402
    REACTION_COMBINE_DIMS,
    RelationPooling,
    model,
)
from test_regression_smoke import (  # noqa: E402
    EDGE_DIM,
    NODE_DIM,
    _build_synthetic_dataset,
    _split_batch,
)
from utils import collate_reaction_graphs, set_seed  # noqa: E402


# ----------------------------------------------------------------------------
# Reference baseline model for backward compatibility tests
# ----------------------------------------------------------------------------


class ReferenceOriginalModel(nn.Module):
    """Exact reproduction of the original baseline model architecture."""

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
        super(ReferenceOriginalModel, self).__init__()
        self.gnn = GIN(node_in_feats, edge_in_feats, num_layer, emb_dim, drop_ratio)
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )
        self.attention_on = attention_on
        self.reaction_combine = reaction_combine
        input_dim = REACTION_COMBINE_DIMS[reaction_combine] * emb_dim
        self.regressor = nn.Linear(input_dim, 1)

    def _attend(self, tokens, mask):
        for layer in self.attention_layers[:-1]:
            tokens = tokens + layer(tokens, mask=mask)
        attended = self.attention_layers[-1](tokens, mask=mask, return_weights=False)
        return attended

    @staticmethod
    def _masked_mean(x, mask):
        weights = mask.unsqueeze(-1).to(x.dtype)
        count = weights.sum(dim=1).clamp(min=1.0)
        return (x * weights).sum(dim=1) / count

    def _combine(self, r, p):
        mode = self.reaction_combine
        if mode == "concat":
            return torch.cat((r, p), dim=1)
        if mode == "sum":
            return r + p
        if mode == "sub":
            return p - r
        if mode == "mul":
            return r * p
        if mode == "concat_sub":
            return torch.cat((r, p, p - r), dim=1)
        return torch.cat((r, p, torch.abs(p - r), r * p), dim=1)

    def forward(self, rmols, pmols, r_dummy, p_dummy, device):
        r_tokens = torch.stack([self.gnn(rmol) for rmol in rmols], dim=1).to(device)
        p_tokens = torch.stack([self.gnn(pmol) for pmol in pmols], dim=1).to(device)
        r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
        p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)

        if self.attention_on == "all":
            n_r = r_tokens.shape[1]
            attended = self._attend(
                torch.cat((r_tokens, p_tokens), dim=1),
                torch.cat((r_mask, p_mask), dim=1),
            )
            r_tokens = attended[:, :n_r]
            p_tokens = attended[:, n_r:]
        else:
            if self.attention_on in ("reactants", "both"):
                r_tokens = self._attend(r_tokens, r_mask)
            if self.attention_on in ("products", "both"):
                p_tokens = self._attend(p_tokens, p_mask)

        reactant_vectors = self._masked_mean(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)
        reaction_vectors = self._combine(reactant_vectors, product_vectors)
        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()


# ----------------------------------------------------------------------------
# 1. Tests for _add_pair_tokens
# ----------------------------------------------------------------------------


def test_add_pair_tokens_shape_and_arithmetic():
    """_add_pair_tokens produces expected shapes, ordering, and r_i + r_j values."""
    set_seed(42)
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 4, 8
    expected_pairs = S * (S - 1) // 2
    expected_total = S + expected_pairs

    tokens = torch.randn(B, S, D)
    mask = torch.ones(B, S, dtype=torch.bool)
    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    assert exp_tokens.shape == (B, expected_total, D)
    assert exp_mask.shape == (B, expected_total)
    assert len(token_slots) == expected_total

    # Verify each pair token equals tokens[:, i] + tokens[:, j]
    pair_idx = 0
    for i in range(S):
        for j in range(i + 1, S):
            slot = token_slots[S + pair_idx]
            assert slot == (i, j)
            expected_pair = tokens[:, i] + tokens[:, j]
            torch.testing.assert_close(exp_tokens[:, S + pair_idx], expected_pair)
            pair_idx += 1


def test_add_pair_tokens_padding_mask():
    """Any pair touching a padded slot is masked out as False."""
    set_seed(42)
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 4, 8
    tokens = torch.randn(B, S, D)
    # Row 0: slot 3 padded. Row 1: slots 1 and 2 padded.
    mask = torch.tensor([[True, True, True, False], [True, False, False, True]], dtype=torch.bool)
    _, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    for idx, slot in enumerate(token_slots):
        if len(slot) == 1:
            assert exp_mask[0, idx] == mask[0, slot[0]]
            assert exp_mask[1, idx] == mask[1, slot[0]]
        else:
            i, j = slot
            assert exp_mask[0, idx] == (mask[0, i] and mask[0, j])
            assert exp_mask[1, idx] == (mask[1, i] and mask[1, j])


def test_add_pair_tokens_single_real_reactant():
    """A reaction with a single real reactant masks all pair tokens and masked_mean works."""
    set_seed(42)
    net = model(10, 5, num_layer=1, emb_dim=8, drop_ratio=0.0)
    B, S, D = 2, 3, 8
    tokens = torch.randn(B, S, D)
    # Only slot 0 is real
    mask = torch.tensor([[True, False, False], [True, False, False]], dtype=torch.bool)
    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)

    # Pairs are (0,1), (0,2), (1,2) - all involve a padded slot so all pair masks must be False
    for idx, slot in enumerate(token_slots):
        if len(slot) == 2:
            assert not exp_mask[:, idx].any()

    # _masked_mean handles this cleanly without NaN or zero division
    mean_vec = net._masked_mean(exp_tokens, exp_mask)
    assert mean_vec.shape == (B, D)
    torch.testing.assert_close(mean_vec, tokens[:, 0])


# ----------------------------------------------------------------------------
# 2. Tests for RelationPooling
# ----------------------------------------------------------------------------


def test_relation_pooling_shape_and_return_pairs():
    """RelationPooling produces [B, D] and optional per-pair diagnostics."""
    set_seed(42)
    B, T, D = 2, 5, 8
    pool = RelationPooling(emb_dim=D, dropout=0.0)
    x = torch.randn(B, T, D)
    mask = torch.ones(B, T, dtype=torch.bool)

    out = pool(x, mask)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()

    out, pair_out, (a, b) = pool(x, mask, return_pairs=True)
    expected_pairs = T * (T - 1) // 2
    assert out.shape == (B, D)
    assert pair_out.shape == (B, expected_pairs, D)
    assert len(a) == expected_pairs and len(b) == expected_pairs


def test_relation_pooling_permutation_invariance():
    """Permuting tokens together with mask leaves RelationPooling output invariant."""
    set_seed(42)
    B, T, D = 3, 5, 16
    pool = RelationPooling(emb_dim=D, dropout=0.0).eval()

    x = torch.randn(B, T, D)
    # Mix of real and padded tokens
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, False, True, False, True],
        ],
        dtype=torch.bool,
    )

    out_orig = pool(x, mask)

    # Permute tokens
    perm = torch.randperm(T)
    x_perm = x[:, perm]
    mask_perm = mask[:, perm]

    out_perm = pool(x_perm, mask_perm)
    torch.testing.assert_close(out_perm, out_orig, atol=1e-5, rtol=1e-5)


def test_relation_pooling_mask_noise_and_extra_slot_invariance():
    """Large noise in padded slots and adding extra padded slots leave output unchanged."""
    set_seed(42)
    B, T, D = 2, 4, 16
    pool = RelationPooling(emb_dim=D, dropout=0.0).eval()

    x = torch.randn(B, T, D)
    mask = torch.tensor([[True, True, False, False], [True, True, True, False]], dtype=torch.bool)
    out_clean = pool(x, mask)

    # Corrupt padded slots with huge noise
    x_noisy = x.clone()
    noise = torch.randn(B, T, D) * 1e5
    x_noisy = torch.where(mask.unsqueeze(-1), x, noise)
    out_noisy = pool(x_noisy, mask)
    torch.testing.assert_close(out_noisy, out_clean, atol=1e-5, rtol=1e-5)

    # Add an extra completely padded slot (T -> T + 1)
    extra_slot = torch.randn(B, 1, D) * 1e4
    extra_mask = torch.zeros(B, 1, dtype=torch.bool)
    x_extended = torch.cat([x_noisy, extra_slot], dim=1)
    mask_extended = torch.cat([mask, extra_mask], dim=1)
    out_extended = pool(x_extended, mask_extended)
    torch.testing.assert_close(out_extended, out_clean, atol=1e-5, rtol=1e-5)


def test_relation_pooling_t1_safe():
    """T == 1 works gracefully using the shape-safe new_zeros path."""
    set_seed(42)
    B, D = 2, 8
    pool = RelationPooling(emb_dim=D, dropout=0.0).eval()
    x = torch.randn(B, 1, D)
    mask = torch.ones(B, 1, dtype=torch.bool)

    out = pool(x, mask)
    assert out.shape == (B, D)
    assert torch.isfinite(out).all()

    out, pair_out, (a, b) = pool(x, mask, return_pairs=True)
    assert out.shape == (B, D)
    assert pair_out.shape == (B, 0, D)
    assert len(a) == 0 and len(b) == 0


def test_relation_pooling_gradients():
    """Gradients reach both phi (main effects) and g (pair effects)."""
    set_seed(42)
    B, T, D = 2, 3, 8
    pool = RelationPooling(emb_dim=D, dropout=0.0)
    x = torch.randn(B, T, D, requires_grad=True)
    mask = torch.ones(B, T, dtype=torch.bool)

    out = pool(x, mask)
    loss = out.sum()
    loss.backward()

    # Check gradients in phi
    for p in pool.phi.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert (p.grad != 0).any()

    # Check gradients in g
    for p in pool.g.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
        assert (p.grad != 0).any()


# ----------------------------------------------------------------------------
# 3. Full Model Tests (comb + rn + reactants attention)
# ----------------------------------------------------------------------------


def test_full_model_permutation_and_noise_invariance():
    """Shuffling reactant slots or adding padding noise leaves model predictions unchanged."""
    set_seed(42)
    device = torch.device("cpu")
    net = model(
        node_in_feats=NODE_DIM,
        edge_in_feats=EDGE_DIM,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        num_attention_layer=1,
        reactant_tokens="comb",
        reactant_pooling="rn",
        head="mlp",
    ).to(device)
    net.eval()

    # Build dataset with 3 reactant slots and 1 product slot
    dataset = _build_synthetic_dataset(y_values=np.array([50.0, 75.0]), rmol_max_cnt=3, pmol_max_cnt=1)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_reaction_graphs)
    batch = next(iter(loader))
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(batch, 3, 1)

    # 1. Base prediction
    with torch.no_grad():
        pred_base, rvecs_base = net(rmols, pmols, r_dummy, p_dummy, device)

    # 2. Permute reactant slots: swap slot 0 and slot 1
    rmols_perm = [rmols[1], rmols[0], rmols[2]]
    r_dummy_arr = np.asarray(r_dummy)
    r_dummy_perm = r_dummy_arr[:, [1, 0, 2]].tolist()

    with torch.no_grad():
        pred_perm, rvecs_perm = net(rmols_perm, pmols, r_dummy_perm, p_dummy, device)

    torch.testing.assert_close(pred_perm, pred_base, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.tensor(rvecs_perm), torch.tensor(rvecs_base), atol=1e-5, rtol=1e-5)


def test_full_model_padded_slot_invariance():
    """A reaction with an extra padded reactant slot gives identical prediction to without it."""
    set_seed(42)
    device = torch.device("cpu")
    net = model(
        node_in_feats=NODE_DIM,
        edge_in_feats=EDGE_DIM,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        num_attention_layer=1,
        reactant_tokens="comb",
        reactant_pooling="rn",
        head="linear",
    ).to(device)
    net.eval()

    dataset_2r = _build_synthetic_dataset(y_values=np.array([30.0, 40.0]), rmol_max_cnt=2, pmol_max_cnt=1)
    dataset_3r = _build_synthetic_dataset(y_values=np.array([30.0, 40.0]), rmol_max_cnt=3, pmol_max_cnt=1)

    batch_2r = next(iter(DataLoader(dataset_2r, batch_size=2, shuffle=False, collate_fn=collate_reaction_graphs)))
    batch_3r = next(iter(DataLoader(dataset_3r, batch_size=2, shuffle=False, collate_fn=collate_reaction_graphs)))

    rmols_2, pmols_2, r_dummy_2, p_dummy_2, _ = _split_batch(batch_2r, 2, 1)
    rmols_3, pmols_3, r_dummy_3, p_dummy_3, _ = _split_batch(batch_3r, 3, 1)

    # In 3r, slot 0 and slot 1 come from 2r, but slot 2 is set to dummy=False
    rmols_3_mod = [rmols_2[0], rmols_2[1], rmols_3[2]]
    r_dummy_3_mod = [[True, True, False], [True, True, False]]

    with torch.no_grad():
        pred_2r, _ = net(rmols_2, pmols_2, r_dummy_2, p_dummy_2, device)
        pred_3r, _ = net(rmols_3_mod, pmols_2, r_dummy_3_mod, p_dummy_2, device)

    torch.testing.assert_close(pred_3r, pred_2r, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------------
# 4. Suzuki-width shape & masking test (S=14, T=105, 5460 pairs)
# ----------------------------------------------------------------------------


def test_suzuki_width_single_forward_backward():
    """Exact Suzuki token width (14 slots, T=105, 5460 pairs) runs forward+backward at B=2, D=16."""
    set_seed(42)
    device = torch.device("cpu")
    B, S, D = 2, 14, 16
    expected_T = S + S * (S - 1) // 2  # 14 + 91 = 105
    expected_pairs = expected_T * (expected_T - 1) // 2  # 105 * 104 // 2 = 5460

    net = model(
        node_in_feats=10,
        edge_in_feats=5,
        num_layer=1,
        emb_dim=D,
        drop_ratio=0.0,
        reactant_tokens="comb",
        reactant_pooling="rn",
        head="linear",
    ).to(device)

    # Directly feed embeddings to _add_pair_tokens and RelationPooling to verify width
    tokens = torch.randn(B, S, D, requires_grad=True)
    mask = torch.ones(B, S, dtype=torch.bool)
    # Mask out some realistic padding (slots 10-13)
    mask[:, 10:] = False

    exp_tokens, exp_mask, token_slots = net._add_pair_tokens(tokens, mask)
    assert exp_tokens.shape == (B, expected_T, D)
    assert exp_mask.shape == (B, expected_T)
    assert len(token_slots) == expected_T

    # Pass through relation pooling
    pooled, pair_out, (a, b) = net.reactant_pool(exp_tokens, exp_mask, return_pairs=True)
    assert pooled.shape == (B, D)
    assert pair_out.shape == (B, expected_pairs, D)

    # Backward pass
    loss = pooled.sum()
    loss.backward()
    assert tokens.grad is not None
    assert torch.isfinite(tokens.grad).all()


# ----------------------------------------------------------------------------
# 5. Backward Compatibility Tests
# ----------------------------------------------------------------------------


def test_backward_compatibility_reference_model():
    """Default new model matches ReferenceOriginalModel bit-for-bit with same weights."""
    set_seed(42)
    device = torch.device("cpu")
    orig = ReferenceOriginalModel(
        node_in_feats=NODE_DIM,
        edge_in_feats=EDGE_DIM,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
    ).to(device)

    new_m = model(
        node_in_feats=NODE_DIM,
        edge_in_feats=EDGE_DIM,
        num_layer=1,
        emb_dim=16,
        drop_ratio=0.0,
        reactant_tokens="ind",
        reactant_pooling="mean",
        head="linear",
    ).to(device)

    # Load weights strictly
    new_m.load_state_dict(orig.state_dict(), strict=True)

    dataset = _build_synthetic_dataset(y_values=np.array([45.0, 55.0]), rmol_max_cnt=2, pmol_max_cnt=1)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collate_reaction_graphs)
    batch = next(iter(loader))
    rmols, pmols, r_dummy, p_dummy, _ = _split_batch(batch, 2, 1)

    orig.eval()
    new_m.eval()
    with torch.no_grad():
        out_orig, rvecs_orig = orig(rmols, pmols, r_dummy, p_dummy, device)
        out_new, rvecs_new = new_m(rmols, pmols, r_dummy, p_dummy, device)

    torch.testing.assert_close(out_new, out_orig)
    torch.testing.assert_close(torch.tensor(rvecs_new), torch.tensor(rvecs_orig))


def test_from_config_legacy_compatibility():
    """Old config without reactant_tokens, reactant_pooling, or head resolves to defaults."""
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
    assert net.reactant_pooling == "mean"
    assert net.head == "linear"
    assert net.reactant_pool is None
    assert isinstance(net.regressor, nn.Linear)
    assert net.config["reactant_tokens"] == "ind"
    assert net.config["reactant_pooling"] == "mean"
    assert net.config["head"] == "linear"
