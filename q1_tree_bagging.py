"""Q1: single-tree baseline, bagging from scratch, OOB / disagreement / margin, masked test.

Run:  python q1_tree_bagging.py | tee results/q1_log.txt
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import common as C

DEPTHS = [2, 4, 8, 16, None]
MIN_LEAF = [1, 2, 4]
B_LIST = [1, 5, 10, 25, 50, 100, 200]
TREE_SEED_TAG, BAG_SEED_TAG = 1, 2


def show(rows, cols):
    print(" | ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>12.4f}" if isinstance(r[c], float) else f"{str(r[c]):>12}" for c in cols))


# ----------------------------------------------------------------------------- (a)
def part_a(D):
    print("\n=== Q1(a) single tree grid (selection on validation only) ===")
    rs = C.child_seed(TREE_SEED_TAG)
    rows = []
    for depth in DEPTHS:
        for msl in MIN_LEAF:
            t = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=msl, random_state=rs)
            t.fit(D["X_train"], D["y_train"])
            rows.append({"max_depth": depth, "min_samples_leaf": msl,
                         "train_acc": C.accuracy(D["y_train"], t.predict(D["X_train"])),
                         "val_acc": C.accuracy(D["y_val"], t.predict(D["X_val"])),
                         **C.leaf_stats(t, D["X_train"])})
    # rule: highest val acc; ties -> fewer leaves (simpler tree); then grid order
    best = max(range(len(rows)), key=lambda i: (rows[i]["val_acc"], -rows[i]["n_leaves"], -i))
    show(rows, ["max_depth", "min_samples_leaf", "actual_depth", "n_leaves", "mean_samples_per_leaf",
                "frac_single_example_leaves", "train_acc", "val_acc"])

    frozen = {"max_depth": rows[best]["max_depth"],
              "min_samples_leaf": rows[best]["min_samples_leaf"], "random_state": rs}
    (C.CONFIG_DIR / "frozen_tree.json").write_text(json.dumps(frozen, indent=2))
    tree = DecisionTreeClassifier(**frozen).fit(D["X_train"], D["y_train"])
    sel = {**rows[best], "test_acc": C.accuracy(D["y_test"], tree.predict(D["X_test"]))}
    print("\nFROZEN tree:", frozen)
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in sel.items()})

    # figure: what changes with depth
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    x = np.arange(len(DEPTHS))
    for msl in MIN_LEAF:
        rr = [r for r in rows if r["min_samples_leaf"] == msl]
        ax[0].plot(x, [r["train_acc"] for r in rr], "--o", label=f"train, msl={msl}")
        ax[0].plot(x, [r["val_acc"] for r in rr], "-s", label=f"val, msl={msl}")
        ax[1].plot(x, [r["n_leaves"] for r in rr], "-o", label=f"msl={msl}")
        ax[2].plot(x, [r["frac_single_example_leaves"] for r in rr], "-o", label=f"msl={msl}")
    rr = [r for r in rows if r["min_samples_leaf"] == 1]
    ax[0].plot(x, [min(r["n_leaves"] / C.N_CLASSES, 1) for r in rr], ":k", label="n_leaves / 40 (msl=1)")
    for a, t in zip(ax, ["Accuracy", "Number of leaves", "Fraction of 1-example leaves"]):
        a.set_xticks(x, [str(d) for d in DEPTHS]); a.set_xlabel("max_depth"); a.set_title(t)
        a.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q1a_depth_sweep.png", dpi=200); plt.close(fig)
    return rows, sel, frozen, tree


# ----------------------------------------------------------------------------- (b) + (c)
def part_bc(D, frozen):
    print("\n=== Q1(b,c) bagging from scratch ===")
    params = {k: frozen[k] for k in ("max_depth", "min_samples_leaf")}
    bag_seed = C.child_seed(BAG_SEED_TAG)
    bag = C.BaggedTrees(max(B_LIST), params, seed=bag_seed).fit(D["X_train"], D["y_train"])
    P = {s: bag.tree_predictions(D[f"X_{s}"]) for s in ("train", "val", "test")}
    n = len(D["y_train"])
    theory = n * (1 - (1 - 1 / n) ** n)
    distinct = np.array([len(np.unique(i)) for i in bag.bootstrap_idx_])

    rows = []
    for B in B_LIST:
        r = {"B": B}
        for s in ("train", "val", "test"):
            r[f"{s}_acc"] = C.accuracy(D[f"y_{s}"], C.votes_from_predictions(P[s][:B]).argmax(1))
        r["distinct_mean"] = float(distinct[:B].mean())
        r["distinct_std"] = float(distinct[:B].std())
        r["distinct_theory"] = float(theory)
        oob = C.votes_from_predictions(P["train"][:B], mask=bag.oob_mask_[:B])
        cov = oob.sum(1) > 0
        r["oob_coverage"] = float(cov.mean())
        r["oob_acc"] = C.accuracy(D["y_train"][cov], oob[cov].argmax(1))
        frac_val = C.votes_from_predictions(P["val"][:B]) / B
        r["disagreement_val"] = C.pairwise_disagreement(P["val"][:B])
        r["margin_val"] = float(C.vote_margin(frac_val).mean())
        rows.append(r)
    show(rows, ["B", "train_acc", "val_acc", "test_acc", "distinct_mean", "distinct_theory"])
    print()
    show(rows, ["B", "oob_acc", "oob_coverage", "disagreement_val", "margin_val"])

    # select B on validation only; ties -> smallest B
    best = max(range(len(rows)), key=lambda i: (rows[i]["val_acc"], -rows[i]["B"]))
    B_final = rows[best]["B"]
    cfg = {"n_estimators": B_final, "seed": bag_seed, "tree_params": params}
    (C.CONFIG_DIR / "frozen_bagging.json").write_text(json.dumps(cfg, indent=2))
    print(f"\nFROZEN bagging: B = {B_final}  (theory distinct = {theory:.2f} of {n})")

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    Bs = [r["B"] for r in rows]
    ax[0].plot(Bs, [r["oob_acc"] for r in rows], "-o", label="OOB acc")
    ax[0].plot(Bs, [r["val_acc"] for r in rows], "-s", label="val acc")
    ax[1].plot(Bs, [r["disagreement_val"] for r in rows], "-o")
    ax[1].set_ylim(0, 1)
    ax[2].plot(Bs[1:], [r["margin_val"] for r in rows[1:]], "-o")
    ax[2].plot(Bs[:1], [rows[0]["margin_val"]], "o", mfc="none", color="tab:blue")
    ax[2].annotate("B=1: trivially 1", (Bs[0], rows[0]["margin_val"]), textcoords="offset points",
                   xytext=(8, -12), fontsize=8)
    ax[2].set_ylim(0, 1.05)
    for a, t in zip(ax, ["OOB accuracy", "Mean pairwise disagreement (val)", "Mean vote margin (val)"]):
        a.set_xscale("log"); a.set_xlabel("B (number of trees)"); a.set_title(t)
        a.axvline(B_final, color="grey", ls=":")
    ax[0].legend()
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q1c_oob_disagreement_margin.png", dpi=200); plt.close(fig)
    return rows, bag, B_final


# ----------------------------------------------------------------------------- (d)
def part_d(D, tree, bag, B_final):
    print("\n=== Q1(d) single tree vs bagging, clean and masked test ===")
    res, per = {}, {}
    for name in ("test", "test_masked"):
        X, y = D[f"X_{name}"], D["y_test"]
        proba = tree.predict_proba(X)
        tp, tconf = tree.classes_[proba.argmax(1)], proba.max(1)
        frac = bag.vote_fractions(X, B_final)
        bp, bconf = frac.argmax(1), frac.max(1)
        per[name] = dict(tp=tp, tconf=tconf, bp=bp, bconf=bconf, frac=frac)
        res[name] = {
            "tree_acc": C.accuracy(y, tp), "tree_mean_conf": float(tconf.mean()),
            "tree_conf_correct": float(tconf[tp == y].mean()) if (tp == y).any() else None,
            "tree_conf_wrong": float(tconf[tp != y].mean()) if (tp != y).any() else None,
            "bag_acc": C.accuracy(y, bp), "bag_mean_conf": float(bconf.mean()),
            "bag_conf_correct": float(bconf[bp == y].mean()) if (bp == y).any() else None,
            "bag_conf_wrong": float(bconf[bp != y].mean()) if (bp != y).any() else None,
        }
        print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in res[name].items()})

    y = D["y_test"]

    def case(name, i, kind):
        p = per[name]; votes = np.round(p["frac"][i] * B_final).astype(int)
        top = np.argsort(-votes)[:5]
        return {"kind": kind, "set": name, "test_pos": int(i), "dataset_idx": int(D["idx_test"][i]),
                "true": int(y[i]), "tree_pred": int(p["tp"][i]), "tree_conf": float(p["tconf"][i]),
                "bag_pred": int(p["bp"][i]), "votes_true": int(votes[y[i]]),
                "votes_tree_pred": int(votes[p["tp"][i]]), "B": B_final,
                "top5_votes": [(int(c), int(votes[c])) for c in top]}

    cases = []
    for name in ("test", "test_masked"):  # prefer clean test set; fall back to masked
        p = per[name]
        fix = np.flatnonzero((p["tp"] != y) & (p["bp"] == y))
        fix = fix[np.argsort(-p["frac"][fix, y[fix]])]           # clearest corrections first
        both = np.flatnonzero((p["tp"] != y) & (p["bp"] != y))
        both = both[np.argsort(-p["bconf"][both])]                 # most confident wrong first
        for kind, arr in (("bagging_corrects", fix), ("both_wrong", both)):
            need = 2 - sum(c["kind"] == kind for c in cases)
            cases += [case(name, i, kind) for i in arr[:max(need, 0)]]
    for c in cases:
        print(c)

    fig, ax = plt.subplots(len(cases), 5, figsize=(13.5, 2.7 * len(cases)), squeeze=False)
    for r, c in enumerate(cases):
        img = D["img_test_masked" if c["set"] == "test_masked" else "img_test"][c["test_pos"]]
        ax[r, 0].imshow(img, cmap="gray")
        ax[r, 0].set_title(f"{c['kind']} ({c['set']})\ntrue {c['true']}, tree {c['tree_pred']}, "
                           f"bag {c['bag_pred']}", fontsize=8)
        cls = [k for k, _ in c["top5_votes"]]
        if c["tree_pred"] not in cls:  # always show the single tree's choice
            cls = cls[:4] + [c["tree_pred"]]
        votes_of = dict(c["top5_votes"]); votes_of[c["tree_pred"]] = c["votes_tree_pred"]
        cnt = [votes_of[k] for k in cls]
        col = ["tab:green" if k == c["true"] else "tab:red" if k == c["tree_pred"] else "tab:grey" for k in cls]
        ax[r, 1].bar([str(k) for k in cls], cnt, color=col)
        ax[r, 1].set_title(f"bagging votes (B={B_final})\ngreen=true, red=tree pred", fontsize=8)
        for j, (cl, lab) in enumerate(((c["true"], "train face, true id"),
                                       (c["tree_pred"], "train face, tree pred"),
                                       (c["bag_pred"], "train face, bagging pred"))):
            ax[r, 2 + j].imshow(D["img_train"][D["y_train"] == cl][0], cmap="gray")
            ax[r, 2 + j].set_title(f"{lab} {cl}", fontsize=8)
        for a in (ax[r, 0], ax[r, 2], ax[r, 3], ax[r, 4]):
            a.axis("off")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q1d_cases.png", dpi=200); plt.close(fig)
    return res, cases


def main():
    D = C.load_data()
    print(f"seed={D['seed']}  train={len(D['y_train'])} val={len(D['y_val'])} test={len(D['y_test'])}")
    grid, sel, frozen, tree = part_a(D)
    bag_rows, bag, B_final = part_bc(D, frozen)
    d_res, cases = part_d(D, tree, bag, B_final)
    out = {"seed": D["seed"], "a_grid": grid, "a_selected": sel, "a_frozen": frozen,
           "bc_rows": bag_rows, "B_final": B_final, "d_results": d_res, "d_cases": cases}
    (C.RESULTS_DIR / "q1.json").write_text(json.dumps(C.to_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
