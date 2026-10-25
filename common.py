
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------------------------------- settings
ROLL_NUMBER = os.environ.get("ROLL_NUMBER", "P25CS0201")
N_CLASSES = 40
N_TRAIN, N_VAL, N_TEST = 6, 2, 2
MASK_ROWS = slice(24, 40)  # rows 24..39 inclusive
MASK_COLS = slice(24, 40)  # cols 24..39 inclusive

ROOT = Path(__file__).resolve().parent
SPLIT_PATH = ROOT / "split" / "split.json"
CONFIG_DIR = ROOT / "configs"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for _d in (SPLIT_PATH.parent, CONFIG_DIR, RESULTS_DIR, FIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def roll_seed(roll: str = ROLL_NUMBER) -> int:
    digits = "".join(ch for ch in roll if ch.isdigit())
    if not digits:
        raise ValueError("Set ROLL_NUMBER (edit common.py or `export ROLL_NUMBER=...`).")
    return int(digits)


SEED = roll_seed()


def child_seed(tag: int) -> int:
   
    return int(np.random.SeedSequence([SEED, tag]).generate_state(1)[0])


# --------------------------------------------------------------------------- data + split
def make_split(y: np.ndarray, seed: int) -> dict:
 
    rng = np.random.default_rng(seed)
    parts = {"train": [], "val": [], "test": []}
    for c in range(N_CLASSES):
        perm = rng.permutation(np.flatnonzero(y == c))
        parts["train"].extend(perm[:N_TRAIN].tolist())
        parts["val"].extend(perm[N_TRAIN:N_TRAIN + N_VAL].tolist())
        parts["test"].extend(perm[N_TRAIN + N_VAL:].tolist())
    return parts


def load_or_create_split(y: np.ndarray) -> dict:
    if SPLIT_PATH.exists():
        split = json.loads(SPLIT_PATH.read_text())
        if split["seed"] != SEED:
            raise RuntimeError(f"{SPLIT_PATH} was made with seed {split['seed']}, not {SEED}.")
        fresh = make_split(y, SEED)  # sanity check: saved file must match the data + seed
        if any(split[k] != fresh[k] for k in ("train", "val", "test")):
            raise RuntimeError(f"{SPLIT_PATH} does not match make_split(y, {SEED}); delete it and rerun.")
        return split
    split = {"roll_number": ROLL_NUMBER, "seed": SEED, **make_split(y, SEED)}
    SPLIT_PATH.write_text(json.dumps(split, indent=1))
    return split


def mask_images(images: np.ndarray) -> np.ndarray:
    out = images.copy()
    out[:, MASK_ROWS, MASK_COLS] = 0.0
    return out


def load_data() -> dict:
 
    faces = fetch_olivetti_faces(shuffle=False)
    images = faces.images.astype(np.float32)
    y = faces.target.astype(int)
    split = load_or_create_split(y)
    D = {"split": split, "seed": SEED}
    for name in ("train", "val", "test"):
        idx = np.asarray(split[name])
        D[f"idx_{name}"] = idx
        D[f"img_{name}"] = images[idx]
        D[f"X_{name}"] = images[idx].reshape(len(idx), -1)
        D[f"y_{name}"] = y[idx]
    D["img_test_masked"] = mask_images(D["img_test"])
    D["X_test_masked"] = D["img_test_masked"].reshape(len(D["idx_test"]), -1)
    D["y_test_masked"] = D["y_test"]
    return D


# --------------------------------------------------------------------------- metrics
def accuracy(y_true, y_pred) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def votes_from_predictions(preds, n_classes=N_CLASSES, weights=None, mask=None):
   
    preds = np.asarray(preds, dtype=int)
    T, N = preds.shape
    w = np.ones(T) if weights is None else np.asarray(weights, dtype=float)
    votes = np.zeros((N, n_classes))
    rows = np.arange(N)
    for t in range(T):
        m = slice(None) if mask is None else mask[t]
        votes[rows[m], preds[t][m]] += w[t]
    return votes


def vote_margin(fractions: np.ndarray) -> np.ndarray:
    s = np.sort(fractions, axis=1)
    return s[:, -1] - s[:, -2]


def pairwise_disagreement(preds, n_classes=N_CLASSES) -> float:
    preds = np.asarray(preds, dtype=int)
    T, N = preds.shape
    if T < 2:
        return float("nan")
    onehot = np.zeros((T, N * n_classes), dtype=np.float32)
    onehot[np.repeat(np.arange(T), N), np.tile(np.arange(N), T) * n_classes + preds.ravel()] = 1
    agree = onehot @ onehot.T  # agree[i, j] = #samples where tree i and tree j agree
    iu = np.triu_indices(T, k=1)
    return float(np.mean(1.0 - agree[iu] / N))


def leaf_stats(tree: DecisionTreeClassifier, X_fit: np.ndarray) -> dict:
    counts = np.bincount(tree.apply(X_fit))
    counts = counts[counts > 0]
    return {
        "n_leaves": int(tree.get_n_leaves()),
        "mean_samples_per_leaf": float(counts.mean()),
        "frac_single_example_leaves": float(np.mean(counts == 1)),
        "actual_depth": int(tree.get_depth()),
    }


# --------------------------------------------------------------------------- bagging (from scratch)
class BaggedTrees:
 
    def __init__(self, n_estimators: int, tree_params: dict, seed: int, n_classes: int = N_CLASSES):
        self.n_estimators = n_estimators
        self.tree_params = dict(tree_params)
        self.seed = seed
        self.n_classes = n_classes

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.trees_, self.bootstrap_idx_ = [], []
        for _ in range(self.n_estimators):
            idx = rng.integers(0, n, size=n)  # sample n with replacement
            tree = DecisionTreeClassifier(**self.tree_params,
                                          random_state=int(rng.integers(0, 2**32 - 1)))
            tree.fit(X[idx], y[idx])
            self.trees_.append(tree)
            self.bootstrap_idx_.append(idx)
        self.oob_mask_ = np.ones((self.n_estimators, n), dtype=bool)
        for b, idx in enumerate(self.bootstrap_idx_):
            self.oob_mask_[b, idx] = False
        return self

    def tree_predictions(self, X, n_trees=None) -> np.ndarray:
        trees = self.trees_[: (n_trees or len(self.trees_))]
        return np.stack([t.predict(X) for t in trees]).astype(int)

    def vote_fractions(self, X, n_trees=None) -> np.ndarray:
        p = self.tree_predictions(X, n_trees)
        return votes_from_predictions(p, self.n_classes) / len(p)

    def predict(self, X, n_trees=None) -> np.ndarray:
        return self.vote_fractions(X, n_trees).argmax(1)  # ties -> lowest class id


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --------------------------------------------------------------------------- mask pixels as feature ids
MASK_FEATURES = np.array([r * 64 + c for r in range(64)[MASK_ROWS] for c in range(64)[MASK_COLS]])


def split_features(tree: DecisionTreeClassifier) -> np.ndarray:
    f = tree.tree_.feature
    return f[f >= 0]


def path_uses_mask(tree: DecisionTreeClassifier, x: np.ndarray) -> bool:
    nodes = tree.decision_path(x.reshape(1, -1)).indices
    feats = tree.tree_.feature[nodes]
    return bool(np.isin(feats[feats >= 0], MASK_FEATURES).any())


# --------------------------------------------------------------------------- SAMME (from scratch)
EPS_FLOOR = 1e-10


class SAMME:
   

    def __init__(self, n_rounds: int, max_depth: int, seed: int, n_classes: int = N_CLASSES):
        self.n_rounds = n_rounds
        self.max_depth = max_depth
        self.seed = seed
        self.n_classes = n_classes

    def fit(self, X, y, round_log=None):
        K = self.n_classes
        n = len(X)
        rng = np.random.default_rng(self.seed)
        w = np.full(n, 1.0 / n)
        self.learners_, self.alphas_, self.errors_ = [], [], []
        self.weight_history_ = [w.copy()]  # weights used to train learner t (index 0 = uniform)
        self.train_pred_history_ = []
        self.stop_reason_ = "completed all rounds"
        for t in range(self.n_rounds):
            h = DecisionTreeClassifier(max_depth=self.max_depth,
                                       random_state=int(rng.integers(0, 2**32 - 1)))
            h.fit(X, y, sample_weight=w)
            pred = h.predict(X).astype(int)
            miss = pred != y
            eps = float(w[miss].sum() / w.sum())
            if eps >= 1.0 - 1.0 / K:
                self.stop_reason_ = (f"round {t + 1}: eps={eps:.4f} >= 1-1/K={1 - 1 / K:.4f} "
                                     f"(not better than random); learner discarded, stopped")
                break
            perfect = eps <= 0.0
            eps_used = max(eps, EPS_FLOOR)
            alpha = float(np.log((1.0 - eps_used) / eps_used) + np.log(K - 1))
            w_unnorm = w * np.exp(alpha * miss)
            w_new = w_unnorm / w_unnorm.sum()
            if round_log is not None:
                round_log(dict(t=t + 1, w_before=w.copy(), pred=pred.copy(), miss=miss.copy(),
                               eps=eps, alpha=alpha, w_unnorm=w_unnorm.copy(), w_after=w_new.copy()))
            self.learners_.append(h)
            self.alphas_.append(alpha)
            self.errors_.append(eps)
            self.train_pred_history_.append(pred)
            w = w_new
            self.weight_history_.append(w.copy())
            if perfect:
                self.stop_reason_ = f"round {t + 1}: perfect weak learner (eps=0); stopped"
                break
        self.alphas_ = np.asarray(self.alphas_)
        self.errors_ = np.asarray(self.errors_)
        self.weight_history_ = np.asarray(self.weight_history_)
        self.train_pred_history_ = np.asarray(self.train_pred_history_)
        return self

    def learner_predictions(self, X, n_rounds=None) -> np.ndarray:
        hs = self.learners_[: (n_rounds or len(self.learners_))]
        return np.stack([h.predict(X) for h in hs]).astype(int)

    def scores(self, X, n_rounds=None) -> np.ndarray:
        p = self.learner_predictions(X, n_rounds)
        return votes_from_predictions(p, self.n_classes, weights=self.alphas_[: len(p)])

    def q(self, X, n_rounds=None) -> np.ndarray:
        """Normalised weighted vote q_k(x) = S_k(x) / sum_t alpha_t."""
        p = self.learner_predictions(X, n_rounds)
        a = self.alphas_[: len(p)]
        return votes_from_predictions(p, self.n_classes, weights=a) / a.sum()

    def predict(self, X, n_rounds=None) -> np.ndarray:
        return self.scores(X, n_rounds).argmax(1)
