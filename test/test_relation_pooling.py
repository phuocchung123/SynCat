"""
Tests of the relation-network reactant pooling and the MLP regression head.

The graph inputs are synthetic (random small graphs with the repository's
node/edge feature widths, and padding slots built like
`preprocess_utils.add_dummy`), so that no rdkit import is needed.
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

SRC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from attention import ReactionSelfAttention  # noqa: E402
from gin import GIN  # noqa: E402
from model import ATTENTION_TARGETS, model  # noqa: E402
from pooling import RelationPooling  # noqa: E402

NODE_DIM = 155  # matches preprocess_utils.add_mol/add_dummy node feature width
EDGE_DIM = 9  # matches preprocess_utils bond features
EMB_DIM = 16


# ----------------------------------------------------------------------------
# copy of `model` before `reactant_pooling`/`head` existed (commit 3af22da);
# the code is unchanged, only the docstrings are shortened
# ----------------------------------------------------------------------------

ORIGINAL_REACTION_COMBINE_DIMS = {
    "concat": 2,
    "sum": 1,
    "sub": 1,
    "mul": 1,
    "concat_sub": 3,
    "interaction": 4,
}
ORIGINAL_ATTENTION_TARGETS = ("reactants", "products", "both", "all", "none")


class OriginalModel(nn.Module):
    """The reaction-yield model as it was before the new options."""

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
        super(OriginalModel, self).__init__()
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
        self.gnn = GIN(
            node_in_feats,
            edge_in_feats,
            num_layer,
            emb_dim,
            drop_ratio,
        )

        if num_attention_layer < 1:
            raise ValueError(
                "num_attention_layer must be at least 1, got %d" % num_attention_layer
            )
        if attention_on not in ORIGINAL_ATTENTION_TARGETS:
            raise ValueError(
                "attention_on must be one of %s, got %r"
                % (list(ORIGINAL_ATTENTION_TARGETS), attention_on)
            )
        self.attention_on = attention_on
        self.attention_layers = nn.ModuleList(
            [
                ReactionSelfAttention(emb_dim, num_heads=num_heads, dropout=drop_ratio)
                for _ in range(num_attention_layer if attention_on != "none" else 0)
            ]
        )

        if reaction_combine not in ORIGINAL_REACTION_COMBINE_DIMS:
            raise ValueError(
                "reaction_combine must be one of %s, got %r"
                % (sorted(ORIGINAL_REACTION_COMBINE_DIMS), reaction_combine)
            )
        self.reaction_combine = reaction_combine
        self.regressor = torch.nn.Linear(
            ORIGINAL_REACTION_COMBINE_DIMS[reaction_combine] * emb_dim, 1
        )

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
        return torch.cat((r, p, torch.abs(p - r), r * p), dim=1)  # interaction

    def forward(
        self,
        rmols: list,
        pmols: list,
        r_dummy: list,
        p_dummy: list,
        device: torch.device,
    ) -> tuple:
        r_tokens = torch.stack([self.gnn(rmol) for rmol in rmols], dim=1).to(device)
        p_tokens = torch.stack([self.gnn(pmol) for pmol in pmols], dim=1).to(device)

        r_mask = torch.as_tensor(np.asarray(r_dummy, dtype=bool), device=device)
        p_mask = torch.as_tensor(np.asarray(p_dummy, dtype=bool), device=device)

        weights = {}
        if self.attention_on == "all":
            n_reactant_slots = r_tokens.shape[1]
            attended, weights["all"] = self._attend(
                torch.cat((r_tokens, p_tokens), dim=1),
                torch.cat((r_mask, p_mask), dim=1),
            )
            r_tokens = attended[:, :n_reactant_slots]
            p_tokens = attended[:, n_reactant_slots:]
        else:
            if self.attention_on in ("reactants", "both"):
                r_tokens, weights["reactants"] = self._attend(r_tokens, r_mask)
            if self.attention_on in ("products", "both"):
                p_tokens, weights["products"] = self._attend(p_tokens, p_mask)

        if self.store_attention:
            self.last_attention = (
                weights if len(weights) > 1 else next(iter(weights.values()), None)
            )

        reactant_vectors = self._masked_mean(r_tokens, r_mask)
        product_vectors = self._masked_mean(p_tokens, p_mask)

        reaction_vectors = self._combine(reactant_vectors, product_vectors)

        out = self.regressor(reaction_vectors).squeeze(-1)
        return out, reaction_vectors.tolist()


# ----------------------------------------------------------------------------
# synthetic reaction inputs
# ----------------------------------------------------------------------------


def _random_graph(rng: np.random.Generator) -> Data:
    """A small ring-shaped molecule-like graph with binary features."""
    n_node = int(rng.integers(2, 6))
    idx = np.arange(n_node)
    src = np.concatenate([idx, np.roll(idx, 1)])
    dst = np.concatenate([np.roll(idx, 1), idx])
    return Data(
        x=torch.from_numpy(rng.random((n_node, NODE_DIM)) > 0.5).float(),
        edge_index=torch.tensor(np.array([src, dst]), dtype=torch.long),
        edge_attr=torch.from_numpy(rng.random((len(src), EDGE_DIM)) > 0.5).float(),
    )


def _dummy_graph() -> Data:
    """A padding slot, as written by `preprocess_utils.add_dummy`."""
    return Data(
        x=torch.zeros((1, NODE_DIM)),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, EDGE_DIM)),
    )


def _make_reactions(real_counts: list, n_slots: int, seed: int = 0) -> dict:
    """
    Per-reaction (graph, is_real) slots: `real_counts[b]` real reactants, then
    padding; one real product each.
    """
    rng = np.random.default_rng(seed)
    reactants = [
        [
            (_random_graph(rng), True) if s < count else (_dummy_graph(), False)
            for s in range(n_slots)
        ]
        for count in real_counts
    ]
    products = [[(_random_graph(rng), True)] for _ in real_counts]
    return {"reactants": reactants, "products": products}


def _collate(reactions: dict) -> tuple:
    """Slot-major batches and masks, the way `collate_reaction_graphs` builds them."""
    reactants, products = reactions["reactants"], reactions["products"]
    rmols = [Batch.from_data_list([g for g, _ in slot]) for slot in zip(*reactants)]
    pmols = [Batch.from_data_list([g for g, _ in slot]) for slot in zip(*products)]
    r_dummy = [[real for _, real in row] for row in reactants]
    p_dummy = [[real for _, real in row] for row in products]
    return rmols, pmols, r_dummy, p_dummy


def _forward(net: nn.Module, reactions: dict) -> tuple:
    rmols, pmols, r_dummy, p_dummy = _collate(reactions)
    return net(rmols, pmols, r_dummy, p_dummy, torch.device("cpu"))


# Suzuki-like padding: 2 to 5 real reactants in 5 slots.
REAL_COUNTS = [3, 5, 2, 4]
N_SLOTS = 5


# ----------------------------------------------------------------------------
# RelationPooling
# ----------------------------------------------------------------------------


def _pooling_inputs(seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    x = torch.randn(len(REAL_COUNTS), N_SLOTS, EMB_DIM)
    mask = torch.tensor(
        [[s < count for s in range(N_SLOTS)] for count in REAL_COUNTS]
    )
    return x, mask


def _pooling(seed: int = 0, dropout: float = 0.1) -> RelationPooling:
    torch.manual_seed(seed)
    return RelationPooling(EMB_DIM, dropout=dropout).eval()


def test_pooling_output_shape():
    x, mask = _pooling_inputs()
    pooled, pair_terms, pair_index = _pooling()(x, mask, return_pairs=True)

    n_pairs = N_SLOTS * (N_SLOTS - 1) // 2
    assert pooled.shape == (len(REAL_COUNTS), EMB_DIM)
    assert pair_terms.shape == (len(REAL_COUNTS), n_pairs, EMB_DIM)
    assert pair_index.shape == (2, n_pairs)
    # every unordered pair exactly once, no self-pairs
    pairs = [tuple(p) for p in pair_index.t().tolist()]
    assert all(i < j for i, j in pairs)
    assert len(set(pairs)) == n_pairs
    # pairs involving a padding slot contribute nothing
    for b, count in enumerate(REAL_COUNTS):
        for k, (i, j) in enumerate(pairs):
            if j >= count:
                assert torch.all(pair_terms[b, k] == 0)


def test_pooling_permutation_invariance():
    x, mask = _pooling_inputs()
    net = _pooling()
    expected = net(x, mask)

    generator = torch.Generator().manual_seed(1)
    for _ in range(5):
        # a different permutation per reaction, applied to slots and mask alike
        order = torch.argsort(torch.rand(x.shape[:2], generator=generator), dim=1)
        x_perm = torch.gather(x, 1, order.unsqueeze(-1).expand_as(x))
        mask_perm = torch.gather(mask, 1, order)
        assert torch.allclose(net(x_perm, mask_perm), expected, atol=1e-5)


def test_pooling_padding_invariance():
    x, mask = _pooling_inputs()
    net = _pooling()
    expected = net(x, mask)

    noisy = x.clone()
    noisy[~mask] = torch.randn(int((~mask).sum()), EMB_DIM) * 1e4
    assert torch.allclose(net(noisy, mask), expected, atol=1e-5)

    # even non-finite values in a padding slot must not leak through
    noisy[~mask] = float("nan")
    assert torch.allclose(net(noisy, mask), expected, atol=1e-5)

    # one more (noisy) padding slot
    extra = torch.randn(x.shape[0], 1, EMB_DIM) * 1e4
    x_more = torch.cat((x, extra), dim=1)
    mask_more = torch.cat((mask, torch.zeros(x.shape[0], 1, dtype=torch.bool)), dim=1)
    assert torch.allclose(net(x_more, mask_more), expected, atol=1e-5)


def test_pooling_single_slot():
    net = _pooling()
    x = torch.randn(3, 1, EMB_DIM)
    mask = torch.ones(3, 1, dtype=torch.bool)
    pooled, pair_terms, pair_index = net(x, mask, return_pairs=True)

    assert pooled.shape == (3, EMB_DIM)
    assert pair_terms.shape == (3, 0, EMB_DIM)
    assert pair_index.shape == (2, 0)
    # no pairs: only the main effect remains
    assert torch.allclose(pooled, net.norm(net.phi(x[:, 0])), atol=1e-6)

    # a reaction without any real compound stays finite
    empty = net(x, torch.zeros(3, 1, dtype=torch.bool))
    assert torch.isfinite(empty).all()


def test_pooling_gradients_reach_phi_and_g():
    x, mask = _pooling_inputs()
    net = _pooling().train()
    # a random projection: the plain sum of a LayerNorm output has ~zero gradient
    weights = torch.randn(EMB_DIM)
    (net(x, mask) * weights).sum().backward()

    for name in ("phi", "g"):
        for layer in (getattr(net, name)[0], getattr(net, name)[3]):
            assert layer.weight.grad is not None, name
            assert torch.isfinite(layer.weight.grad).all(), name
            assert layer.weight.grad.abs().sum() > 0, name


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

ORIGINAL_SETTINGS = [
    {},
    {"attention_on": "all"},
    {"attention_on": "both", "num_heads": 2, "num_attention_layer": 2},
    {"attention_on": "products", "reaction_combine": "interaction"},
    {"attention_on": "none", "reaction_combine": "concat_sub"},
]


@pytest.mark.parametrize("settings", ORIGINAL_SETTINGS)
def test_defaults_reproduce_original_model(settings):
    reactions = _make_reactions(REAL_COUNTS, N_SLOTS)

    torch.manual_seed(0)
    original = OriginalModel(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)
    torch.manual_seed(0)
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)

    # same seed, same initial weights; same parameter names
    original_state, state = original.state_dict(), net.state_dict()
    assert list(state) == list(original_state)
    for key in state:
        assert torch.equal(state[key], original_state[key]), key

    # same weights loaded from the original, same outputs
    torch.manual_seed(1)
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)
    net.load_state_dict(original_state)
    for training in (False, True):
        original.train(training)
        net.train(training)
        torch.manual_seed(2)  # identical dropout masks in training mode
        expected_pred, expected_vectors = _forward(original, reactions)
        torch.manual_seed(2)
        pred, vectors = _forward(net, reactions)
        assert torch.equal(pred, expected_pred)
        assert vectors == expected_vectors


def test_config_without_new_keys_loads():
    torch.manual_seed(0)
    original = OriginalModel(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1)
    assert "reactant_pooling" not in original.config and "head" not in original.config

    net = model.from_config(original.config)
    assert net.config["reactant_pooling"] == "mean"
    assert net.config["head"] == "linear"
    assert net.relation_pooling is None
    assert isinstance(net.regressor, nn.Linear)
    net.load_state_dict(original.state_dict())  # strict: identical parameters

    # a config that has them round-trips
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, reactant_pooling="rn", head="mlp")
    assert model.from_config(net.config).config == net.config


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="reactant_pooling"):
        model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, reactant_pooling="max")
    with pytest.raises(ValueError, match="head"):
        model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, head="deep")


def test_rn_leaves_other_initial_weights_unchanged():
    torch.manual_seed(0)
    mean_net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1)
    torch.manual_seed(0)
    rn_net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, reactant_pooling="rn")

    rn_state = rn_net.state_dict()
    for key, value in mean_net.state_dict().items():
        assert torch.equal(rn_state[key], value), key
    assert any(key.startswith("relation_pooling.") for key in rn_state)


def test_mlp_head():
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, head="mlp").eval()
    assert isinstance(net.regressor, nn.Sequential)
    assert net.regressor[0].in_features == 2 * EMB_DIM
    assert net.regressor[0].out_features == EMB_DIM
    assert net.regressor[-1].out_features == 1

    pred, vectors = _forward(net, _make_reactions(REAL_COUNTS, N_SLOTS))
    assert pred.shape == (len(REAL_COUNTS),)
    assert np.asarray(vectors).shape == (len(REAL_COUNTS), 2 * EMB_DIM)


@pytest.mark.parametrize("attention_on", ATTENTION_TARGETS)
@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_rn_model_invariant_to_reactant_order(attention_on, head):
    torch.manual_seed(0)
    net = model(
        NODE_DIM,
        EDGE_DIM,
        2,
        EMB_DIM,
        0.1,
        attention_on=attention_on,
        reactant_pooling="rn",
        head=head,
    ).eval()
    reactions = _make_reactions(REAL_COUNTS, N_SLOTS)
    with torch.no_grad():
        expected, _ = _forward(net, reactions)

        rng = np.random.default_rng(3)
        for _ in range(3):
            # shuffle the real reactants of every reaction, padding stays at the end
            shuffled = []
            for row, count in zip(reactions["reactants"], REAL_COUNTS):
                order = list(rng.permutation(count)) + list(range(count, N_SLOTS))
                shuffled.append([row[k] for k in order])
            pred, _ = _forward(net, dict(reactions, reactants=shuffled))
            assert torch.allclose(pred, expected, atol=1e-5)

            # one permutation of all slots: padding moves in front, with its mask
            order = rng.permutation(N_SLOTS)
            mixed = [[row[k] for k in order] for row in reactions["reactants"]]
            pred, _ = _forward(net, dict(reactions, reactants=mixed))
            assert torch.allclose(pred, expected, atol=1e-5)


def test_rn_model_stores_pair_terms():
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, reactant_pooling="rn").eval()
    net.store_attention = True
    with torch.no_grad():
        _forward(net, _make_reactions(REAL_COUNTS, N_SLOTS))

    n_pairs = N_SLOTS * (N_SLOTS - 1) // 2
    terms = net.last_pair_terms
    assert np.asarray(terms["norms"]).shape == (len(REAL_COUNTS), n_pairs)
    assert len(terms["pairs"]) == n_pairs
    for b, count in enumerate(REAL_COUNTS):
        for k, (i, j) in enumerate(terms["pairs"]):
            assert (terms["norms"][b][k] > 0) == (j < count)
    assert net.last_attention is not None

    # nothing is stored when the flag is off
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, reactant_pooling="rn").eval()
    with torch.no_grad():
        _forward(net, _make_reactions(REAL_COUNTS, N_SLOTS))
    assert net.last_pair_terms is None
