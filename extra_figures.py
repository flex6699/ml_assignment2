"""

Run:  python extra_figures.py | tee results/extra_log.txt
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter
from sklearn.tree import DecisionTreeClassifier

import common as C


def split_map(trees, weights=None):
  
    w = np.ones(len(trees)) if weights is None else np.asarray(weights, float)
    m = np.zeros(64 * 64)
    for t, wt in zip(trees, w):
        m += wt * np.bincount(C.split_features(t), minlength=64 * 64)
    return (m / m.sum()).reshape(64, 64)


def margins(P):
    s = np.sort(P, axis=1)
    return s[:, -1] - s[:, -2]


def main():
    D = C.load_data()
    ft = json.loads((C.CONFIG_DIR / "frozen_tree.json").read_text())
    fb = json.loads((C.CONFIG_DIR / "frozen_bagging.json").read_text())
    fbo = json.loads((C.CONFIG_DIR / "frozen_boosting.json").read_text())
    tree = DecisionTreeClassifier(**ft).fit(D["X_train"], D["y_train"])
    bag = C.BaggedTrees(fb["n_estimators"], fb["tree_params"], fb["seed"]).fit(D["X_train"], D["y_train"])
    boost = C.SAMME(fbo["n_rounds"], fbo["max_depth"], fbo["seed"]).fit(D["X_train"], D["y_train"])
    y = D["y_test"]

    # ---------------------------------------------------------------- figure A: where models split
    maps = {"single tree": split_map([tree]), f"bagging ({len(bag.trees_)} trees)": split_map(bag.trees_),
            "boosting (alpha-weighted)": split_map(boost.learners_, boost.alphas_)}
    mean_face = D["img_train"].mean(0)
    fig, ax = plt.subplots(1, 3, figsize=(10.5, 3.6))
    for a, (name, m) in zip(ax, maps.items()):
        a.imshow(mean_face, cmap="gray")
        sm = gaussian_filter(m, 1.2)
        a.imshow(np.ma.masked_less(sm, sm.max() * 0.05), cmap="inferno", alpha=0.75)
        a.add_patch(Rectangle((23.5, 23.5), 16, 16, fill=False, ec="cyan", lw=1.5, ls="--"))
        inside = m[C.MASK_ROWS, C.MASK_COLS].sum()
        a.set_title(f"{name}\nsplit weight inside mask: {inside:.1%}", fontsize=9)
        a.axis("off")
    fig.suptitle("Pixels tested by split nodes (on the mean training face); dashed: masked block", fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "extra_split_locations.png", dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- figure B: collapse + margins
    P = {}
    for name in ("test", "test_masked"):
        X = D[f"X_{name}"]
        P[("tree", name)] = tree.predict_proba(X)
        P[("bagging", name)] = bag.vote_fractions(X)
        P[("boosting", name)] = boost.q(X)
    colors = {"tree": "tab:red", "bagging": "tab:blue", "boosting": "tab:green"}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ranks = np.arange(1, 11)
    numbers = {}
    for mdl in ("tree", "bagging", "boosting"):
        for name, ls in (("test", "--"), ("test_masked", "-")):
            pred = P[(mdl, name)].argmax(1)
            share = np.sort(np.bincount(pred, minlength=C.N_CLASSES))[::-1][:10] / len(pred)
            ax[0].plot(ranks, share, ls, marker="o", ms=3, c=colors[mdl],
                       label=f"{mdl}, {'masked' if 'masked' in name else 'clean'}")
    ax[0].axhline(2 / 80, c="grey", ls=":", lw=1, label="ideal (2 faces per identity)")
    ax[0].set_xlabel("rank of predicted identity"); ax[0].set_ylabel("share of test predictions")
    ax[0].set_title("Prediction collapse: most frequent predicted identities", fontsize=9)
    ax[0].legend(fontsize=7)

    bins = np.linspace(0, 1, 51)
    for mdl in ("bagging", "boosting"):
        for name, ls in (("test", "--"), ("test_masked", "-")):
            mg = margins(P[(mdl, name)])
            ax[1].hist(mg, bins=bins, histtype="step", ls=ls, lw=1.6, color=colors[mdl],
                       label=f"{mdl}, {'masked' if 'masked' in name else 'clean'} (median {np.median(mg):.3f})")
            numbers[(mdl, name)] = float(np.median(mg))
    ax[1].set_xlabel("vote margin (winner minus runner-up)"); ax[1].set_ylabel("test faces")
    ax[1].set_title("Vote margins on the 80 test faces", fontsize=9)
    ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "extra_collapse_margins.png", dpi=200); plt.close(fig)

    # ---------------------------------------------------------------- numbers for the report
    tree_m = margins(P[("tree", "test")])
    inside = {k: float(m[C.MASK_ROWS, C.MASK_COLS].sum()) for k, m in maps.items()}
    vals = {
        "BagMarginClean": numbers[("bagging", "test")], "BagMarginMasked": numbers[("bagging", "test_masked")],
        "BoostMarginClean": numbers[("boosting", "test")], "BoostMarginMasked": numbers[("boosting", "test_masked")],
        "TreeMarginOneShare": float(np.mean(tree_m > 0.999)),
    }
    print("median vote margins:", {k: round(v, 4) for k, v in vals.items()})
    print("split weight inside the mask:", {k: round(v, 4) for k, v in inside.items()})
    lines = [f"\\newcommand{{\\{k}}}{{{v:.3f}}}" for k, v in vals.items()]
    (C.RESULTS_DIR / "extra_numbers.tex").write_text("\n".join(lines) + "\n")
    print("wrote figures/extra_split_locations.png, figures/extra_collapse_margins.png, results/extra_numbers.tex")


if __name__ == "__main__":
    main()
