"""
Tests of the relation-network reactant pooling (`--reactant_pooling rn`), the
MLP regression head (`--head mlp`), and of "option A": the relation network
applied directly to the GIN outputs (`--attention_on none`).

The model-level inputs are built with the repository's own featurization
(`reaction_data.get_graph_data`, as in `predict.py`) from a few small
reactions; those tests are skipped when RDKit cannot be imported.
"""

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
from gin import GIN  # noqa: E402
from model import ATTENTION_TARGETS, RelationPooling, model  # noqa: E402
from utils import collate_reaction_graphs  # noqa: E402

try:
    from reaction_data import get_graph_data  # noqa: E402

    RDKIT_ERROR = None
except ImportError as error:  # e.g. the RDKit DLLs are blocked on this machine
    RDKIT_ERROR = str(error)

needs_rdkit = pytest.mark.skipif(
    RDKIT_ERROR is not None, reason="rdkit is not importable: %s" % RDKIT_ERROR
)

NODE_DIM = 155  # matches preprocess_utils.add_mol/add_dummy node feature width
EDGE_DIM = 9  # matches preprocess_utils bond features
EMB_DIM = 16

# 2 to 5 reactants (reagents included), so the batch holds padding slots.
REACTIONS = [
    "CCO.CC(=O)O.OS(=O)(=O)O>>CCOC(C)=O",
    "Brc1ccccc1.OB(O)c1ccccc1.[Pd].[K+].[OH-]>>c1ccc(-c2ccccc2)cc1",
    "CC(=O)Cl.NCc1ccccc1>>CC(=O)NCc1ccccc1",
    "O=Cc1ccccc1.[Na+].[BH4-].CO>>OCc1ccccc1",
]
N_REACTANTS = [3, 5, 2, 4]
N_SLOTS = 5


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
# inputs
# ----------------------------------------------------------------------------


def _featurize(rsmis: list, n_slots: int = N_SLOTS) -> tuple:
    """
    Featurizes reactions with the repository's pipeline and collates them into
    one batch, as `predict.py` does.

    Returns
    -------
    tuple
        (rmols, pmols, r_dummy, p_dummy), the model's inputs.
    """
    pmol_max_cnt = max(r.split(">>")[1].count(".") + 1 for r in rsmis)
    rmol, pmol, reaction = get_graph_data(rsmis, n_slots, pmol_max_cnt)
    dataset = GraphDataset(rmol=rmol, pmol=pmol, reaction=reaction)
    batch = collate_reaction_graphs([dataset[k] for k in range(len(dataset))])
    rmols = list(batch[:n_slots])
    pmols = list(batch[n_slots: n_slots + pmol_max_cnt])
    return rmols, pmols, batch[-4], batch[-3]


def _shuffle_reactants(rsmi: str, rng: np.random.Generator) -> str:
    """The same reaction with its reactants written in a random order."""
    reactants, products = rsmi.split(">>")
    names = reactants.split(".")
    return ".".join(names[k] for k in rng.permutation(len(names))) + ">>" + products


def _forward(net: nn.Module, inputs: tuple) -> tuple:
    return net(*inputs, torch.device("cpu"))


def _build(seed: int = 0, **settings) -> model:
    torch.manual_seed(seed)
    return model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)


# ----------------------------------------------------------------------------
# RelationPooling
# ----------------------------------------------------------------------------


def _pooling_inputs(seed: int = 0) -> tuple:
    torch.manual_seed(seed)
    x = torch.randn(len(N_REACTANTS), N_SLOTS, EMB_DIM)
    mask = torch.tensor([[s < n for s in range(N_SLOTS)] for n in N_REACTANTS])
    return x, mask


def _pooling(seed: int = 0) -> RelationPooling:
    torch.manual_seed(seed)
    return RelationPooling(EMB_DIM, 0.1).eval()


def test_pooling_output_shape():
    x, mask = _pooling_inputs()
    pooled, pair_terms, (i, j) = _pooling()(x, mask, return_pairs=True)

    n_pairs = N_SLOTS * (N_SLOTS - 1) // 2
    assert pooled.shape == (len(N_REACTANTS), EMB_DIM)
    assert pair_terms.shape == (len(N_REACTANTS), n_pairs, EMB_DIM)
    assert i.shape == j.shape == (n_pairs,)
    # every unordered pair exactly once, no self-pairs
    pairs = list(zip(i.tolist(), j.tolist()))
    assert all(a < b for a, b in pairs)
    assert len(set(pairs)) == n_pairs
    # pairs involving a padding slot contribute nothing
    for row, n in enumerate(N_REACTANTS):
        for k, (a, b) in enumerate(pairs):
            if b >= n:
                assert torch.all(pair_terms[row, k] == 0)


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

    # one more (noisy) padding slot
    extra = torch.randn(x.shape[0], 1, EMB_DIM) * 1e4
    x_more = torch.cat((x, extra), dim=1)
    mask_more = torch.cat((mask, torch.zeros(x.shape[0], 1, dtype=torch.bool)), dim=1)
    assert torch.allclose(net(x_more, mask_more), expected, atol=1e-5)


def test_pooling_single_slot():
    net = _pooling()
    x = torch.randn(3, 1, EMB_DIM)
    mask = torch.ones(3, 1, dtype=torch.bool)
    pooled, pair_terms, (i, j) = net(x, mask, return_pairs=True)

    assert pooled.shape == (3, EMB_DIM)
    assert pair_terms.shape == (3, 0, EMB_DIM)
    assert i.numel() == j.numel() == 0
    # no pairs: only the main effect remains
    assert torch.allclose(pooled, net.norm(net.phi(x[:, 0])), atol=1e-6)

    # a reaction without any real compound stays finite
    assert torch.isfinite(net(x, torch.zeros(3, 1, dtype=torch.bool))).all()


def test_pooling_gradients_reach_phi_and_g():
    x, mask = _pooling_inputs()
    net = _pooling().train()
    # a random projection: the plain sum of a LayerNorm output has ~zero gradient
    (net(x, mask) * torch.randn(EMB_DIM)).sum().backward()

    for name in ("phi", "g"):
        for layer in (getattr(net, name)[0], getattr(net, name)[3]):
            assert layer.weight.grad is not None, name
            assert torch.isfinite(layer.weight.grad).all(), name
            assert layer.weight.grad.abs().sum() > 0, name


# ----------------------------------------------------------------------------
# option A: attention_on="none", reactant_pooling="rn"
# ----------------------------------------------------------------------------


@needs_rdkit
def test_option_a_pools_the_raw_gin_outputs():
    net = _build(attention_on="none", reactant_pooling="rn").eval()
    gin_outputs, pool_inputs = [], []
    net.gnn.register_forward_hook(lambda module, args, out: gin_outputs.append(out))
    net.reactant_pool.register_forward_pre_hook(
        lambda module, args: pool_inputs.append(args[0])
    )
    inputs = _featurize(REACTIONS)
    with torch.no_grad():
        _forward(net, inputs)

    # the GIN runs once per reactant slot, then once per product slot
    assert len(gin_outputs) == N_SLOTS + len(inputs[1])
    assert len(pool_inputs) == 1
    assert torch.equal(pool_inputs[0], torch.stack(gin_outputs[:N_SLOTS], dim=1))


@needs_rdkit
def test_option_a_calls_no_attention(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("an attention module was called")

    monkeypatch.setattr(ReactionSelfAttention, "forward", fail)
    monkeypatch.setattr(model, "_attend", fail)
    net = _build(attention_on="none", reactant_pooling="rn").eval()

    assert len(net.attention_layers) == 0
    assert not any(isinstance(m, ReactionSelfAttention) for m in net.modules())
    with torch.no_grad():
        pred, _ = _forward(net, _featurize(REACTIONS))
    assert pred.shape == (len(REACTIONS),)
    assert torch.isfinite(pred).all()


@needs_rdkit
@pytest.mark.parametrize("attention_on", ATTENTION_TARGETS)
@pytest.mark.parametrize("head", ["linear", "mlp"])
def test_rn_prediction_invariant_to_reactant_order(attention_on, head):
    net = _build(attention_on=attention_on, reactant_pooling="rn", head=head).eval()
    inputs = _featurize(REACTIONS)
    with torch.no_grad():
        expected, _ = _forward(net, inputs)

        rng = np.random.default_rng(3)
        for _ in range(3):
            # the reactants written in another order, featurized again
            shuffled = [_shuffle_reactants(r, rng) for r in REACTIONS]
            pred, _ = _forward(net, _featurize(shuffled))
            assert torch.allclose(pred, expected, atol=1e-5)

            # one permutation of all slots: padding moves in front, with its mask
            order = rng.permutation(N_SLOTS)
            rmols, pmols, r_dummy, p_dummy = inputs
            permuted = (
                [rmols[k] for k in order],
                pmols,
                [[row[k] for k in order] for row in r_dummy],
                p_dummy,
            )
            pred, _ = _forward(net, permuted)
            assert torch.allclose(pred, expected, atol=1e-5)


@needs_rdkit
@pytest.mark.parametrize("attention_on", ATTENTION_TARGETS)
def test_rn_prediction_invariant_to_extra_padding(attention_on):
    # Padding slots hold the (non-zero) GIN embedding of a dummy graph; a leak
    # would be invariant to slot permutations, but not to their number.
    net = _build(attention_on=attention_on, reactant_pooling="rn").eval()
    with torch.no_grad():
        expected, _ = _forward(net, _featurize(REACTIONS))
        pred, _ = _forward(net, _featurize(REACTIONS, n_slots=N_SLOTS + 2))
    assert torch.allclose(pred, expected, atol=1e-5)


@needs_rdkit
def test_option_a_gradients_reach_the_gin():
    net = _build(attention_on="none", reactant_pooling="rn").train()
    pred, _ = _forward(net, _featurize(REACTIONS))
    ((pred - 0.5) ** 2).mean().backward()

    for name, parameter in net.gnn.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for name, parameter in net.reactant_pool.named_parameters():
        assert parameter.grad is not None, name


# ----------------------------------------------------------------------------
# backward compatibility
# ----------------------------------------------------------------------------

ORIGINAL_SETTINGS = [
    {},
    {"attention_on": "none"},
    {"attention_on": "all"},
    {"attention_on": "both", "num_heads": 2, "num_attention_layer": 2},
    {"attention_on": "products", "reaction_combine": "interaction"},
]


@needs_rdkit
@pytest.mark.parametrize("settings", ORIGINAL_SETTINGS)
def test_defaults_reproduce_original_model(settings):
    inputs = _featurize(REACTIONS)
    torch.manual_seed(0)
    original = OriginalModel(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)
    net = _build(seed=1, **settings)  # other weights, replaced below
    net.load_state_dict(original.state_dict())

    for training in (False, True):
        original.train(training)
        net.train(training)
        torch.manual_seed(2)  # identical dropout masks in training mode
        expected_pred, expected_vectors = _forward(original, inputs)
        torch.manual_seed(2)
        pred, vectors = _forward(net, inputs)
        assert torch.equal(pred, expected_pred)
        assert vectors == expected_vectors


@pytest.mark.parametrize("settings", ORIGINAL_SETTINGS)
def test_original_state_dict_loads_strictly(settings):
    torch.manual_seed(0)
    original = OriginalModel(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)
    torch.manual_seed(0)
    net = model(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, **settings)

    # same parameter names, and the same seed gives the same initial weights
    assert list(net.state_dict()) == list(original.state_dict())
    for key, value in original.state_dict().items():
        assert torch.equal(net.state_dict()[key], value), key
    net.load_state_dict(original.state_dict(), strict=True)


def test_config_without_new_keys_loads():
    original = OriginalModel(NODE_DIM, EDGE_DIM, 2, EMB_DIM, 0.1, attention_on="none")
    assert "reactant_pooling" not in original.config and "head" not in original.config

    net = model.from_config(original.config)
    assert net.config["reactant_pooling"] == "mean"
    assert net.config["head"] == "linear"
    assert net.config["attention_on"] == "none"
    assert net.reactant_pool is None
    assert isinstance(net.regressor, nn.Linear)
    net.load_state_dict(original.state_dict(), strict=True)

    # a config that has them round-trips
    net = _build(attention_on="none", reactant_pooling="rn", head="mlp")
    assert model.from_config(net.config).config == net.config


def test_rn_leaves_the_other_initial_weights_unchanged():
    mean_state = _build(attention_on="none").state_dict()
    rn_state = _build(attention_on="none", reactant_pooling="rn").state_dict()
    for key, value in mean_state.items():
        assert torch.equal(rn_state[key], value), key
    assert any(key.startswith("reactant_pool.") for key in rn_state)


def test_invalid_options_raise():
    with pytest.raises(ValueError, match="reactant_pooling"):
        _build(reactant_pooling="max")
    with pytest.raises(ValueError, match="head"):
        _build(head="deep")


def test_mlp_head():
    net = _build(head="mlp", reaction_combine="concat_sub")
    assert isinstance(net.regressor, nn.Sequential)
    assert net.regressor[0].in_features == 3 * EMB_DIM
    assert net.regressor[0].out_features == EMB_DIM
    assert isinstance(net.regressor[2], nn.Dropout) and net.regressor[2].p == 0.1
    assert net.regressor[-1].out_features == 1


@needs_rdkit
def test_rn_stores_pair_terms():
    net = _build(attention_on="none", reactant_pooling="rn").eval()
    net.store_attention = True
    with torch.no_grad():
        _forward(net, _featurize(REACTIONS))

    n_pairs = N_SLOTS * (N_SLOTS - 1) // 2
    terms = net.last_pair_terms
    assert np.asarray(terms["norms"]).shape == (len(REACTIONS), n_pairs)
    assert len(terms["pairs"]) == n_pairs
    for row, n in enumerate(N_REACTANTS):
        for k, (i, j) in enumerate(terms["pairs"]):
            assert (terms["norms"][row][k] > 0) == (j < n)

    # nothing is stored when the flag is off
    net = _build(attention_on="none", reactant_pooling="rn").eval()
    with torch.no_grad():
        _forward(net, _featurize(REACTIONS))
    assert net.last_pair_terms is None


@needs_rdkit
def test_cli_builds_option_a():
    from finetune import _build_model
    from main_finetune import build_parser

    defaults = build_parser().parse_args([])
    assert (defaults.attention_on, defaults.reactant_pooling, defaults.head) == (
        "reactants",
        "mean",
        "linear",
    )
    args = build_parser().parse_args(
        ["--attention_on", "none", "--reactant_pooling", "rn", "--head", "mlp",
         "--emb_dim", str(EMB_DIM), "--layer", "2"]
    )
    net = _build_model(args, NODE_DIM, EDGE_DIM)
    assert len(net.attention_layers) == 0
    assert isinstance(net.reactant_pool, RelationPooling)
    assert isinstance(net.regressor, nn.Sequential)
