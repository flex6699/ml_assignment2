"""Q3: bias^2 and variance as measured quantities over 30 within-identity resampled training sets.

"""
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeClassifier

import common as C

R = 30                      # number of resampled training sets
B_DEPTH_STUDY = 100         # bagged ensemble size used in Q3(b)
RESAMPLE_SEED_TAG = 10
N_JOBS = int(os.environ.get("Q3_JOBS", "-1"))
CACHE = C.RESULTS_DIR / "q3_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def show(rows, cols):
    print(" | ".join(f"{c:>13}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>13.4f}" if isinstance(r[c], float) else f"{str(r[c]):>13}" for c in cols))


# ----------------------------------------------------------------------------- resampling
def make_resamples(y_train):
    """R training sets: within each identity, 6 draws with replacement from its 6 training faces."""
    rng = np.random.default_rng(C.child_seed(RESAMPLE_SEED_TAG))
    out = []
    for _ in range(R):
        idx = []
        for c in range(C.N_CLASSES):
            pos = np.flatnonzero(y_train == c)
            idx.extend(rng.choice(pos, size=len(pos), replace=True).tolist())
        out.append(idx)
    return out


# ----------------------------------------------------------------------------- one refit (runs in a worker)
def full_proba(tree, X):
    P = np.zeros((len(X), C.N_CLASSES))
    P[:, tree.classes_] = tree.predict_proba(X)
    return P


def leaf_occupancy(tree, X_fit, orig_ids):
    """Leaf stats on the data the tree was fitted on; 'single' = leaf holding one distinct face."""
    leaves = tree.apply(X_fit)
    ul = np.unique(leaves)
    counts = np.array([(leaves == l).sum() for l in ul])
    distinct = np.array([len(np.unique(orig_ids[leaves == l])) for l in ul])
    return {"n_leaves": len(ul), "mean_samples_per_leaf": counts.mean(),
            "frac_single_face_leaves": float(np.mean(distinct == 1)), "depth": tree.get_depth()}


def fit_and_eval(spec, X, y, orig_ids, X_evals):
    if spec["kind"] == "tree":
        m = DecisionTreeClassifier(**spec["params"], random_state=spec["random_state"]).fit(X, y)
        P = [full_proba(m, Xe) for Xe in X_evals]
        stats = leaf_occupancy(m, X, orig_ids)
    elif spec["kind"] == "bag":
        m = C.BaggedTrees(spec["B"], spec["params"], spec["seed"]).fit(X, y)
        P = [m.vote_fractions(Xe) for Xe in X_evals]
        per_tree = [leaf_occupancy(t, X[b], orig_ids[b]) for t, b in zip(m.trees_, m.bootstrap_idx_)]
        stats = {k: float(np.mean([s[k] for s in per_tree])) for k in per_tree[0]}
        stats["disagreement_test"] = C.pairwise_disagreement(m.tree_predictions(X_evals[0]))
    else:  # boosting
        m = C.SAMME(spec["n_rounds"], spec["max_depth"], spec["seed"]).fit(X, y)
        P = [m.q(Xe) for Xe in X_evals]
        stats = {"learners": len(m.learners_)}
    return P, stats


def run_config(name, spec, D, resamples):
    path = CACHE / f"{name}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=True)
        if z["P_clean"].shape[0] == R:
            print(f"  {name}: loaded from cache")
            return z["P_clean"], z["P_masked"], list(z["stats"])
    t0 = time.perf_counter()
    X_evals = [D["X_test"], D["X_test_masked"]]
    out = Parallel(n_jobs=N_JOBS)(
        delayed(fit_and_eval)(spec, D["X_train"][idx], D["y_train"][idx], np.asarray(idx), X_evals)
        for idx in resamples)
    P_clean = np.stack([o[0][0] for o in out]); P_masked = np.stack([o[0][1] for o in out])
    stats = [o[1] for o in out]
    np.savez_compressed(path, P_clean=P_clean, P_masked=P_masked, stats=np.array(stats, dtype=object))
    print(f"  {name}: {R} refits in {time.perf_counter() - t0:.1f} s")
    return P_clean, P_masked, stats


# ----------------------------------------------------------------------------- measured bias / variance
def bias_variance(P, y):
    """P: (R, N, K) class vectors over repetitions; y: (N,) labels."""
    E = np.eye(C.N_CLASSES)[y]
    pbar = P.mean(0)
    bias2 = float(np.mean(((pbar - E) ** 2).sum(1)))
    var = float(np.mean(((P - pbar) ** 2).sum(2)))
    exp_err = float(np.mean(((P - E) ** 2).sum(2)))  # = bias2 + var exactly
    acc = (P.argmax(2) == y).mean(1)
    return {"bias2": bias2, "variance": var, "expected_sq_error": exp_err,
            "acc_mean": float(acc.mean()), "acc_std": float(acc.std(ddof=1))}


def mean_stats(stats):
    keys = [k for k in stats[0] if isinstance(stats[0][k], (int, float, np.floating, np.integer))]
    return {k: float(np.mean([s[k] for s in stats])) for k in keys}


def main():
    D = C.load_data()
    y = D["y_test"]
    ft = json.loads((C.CONFIG_DIR / "frozen_tree.json").read_text())
    fb = json.loads((C.CONFIG_DIR / "frozen_bagging.json").read_text())
    fbo = json.loads((C.CONFIG_DIR / "frozen_boosting.json").read_text())
    q2 = json.loads((C.RESULTS_DIR / "q2.json").read_text())
    resamples = make_resamples(D["y_train"])
    (C.RESULTS_DIR / "q3_resamples.json").write_text(json.dumps(resamples))
    print(f"seed={D['seed']}  R={R} resampled training sets, "
          f"mean distinct faces per set = {np.mean([len(set(r)) for r in resamples]):.1f} of 240")

    tree_params = {"max_depth": ft["max_depth"], "min_samples_leaf": ft["min_samples_leaf"]}
    rs = ft["random_state"]
    msl = ft["min_samples_leaf"]

    # ---------------------------------------------------------------- (a) selected models
    print("\n=== Q3(a) selected single tree, final bagging, selected boosting (no re-tuning) ===")
    specs_a = {
        "tree_selected": {"kind": "tree", "params": tree_params, "random_state": rs},
        "bag_selected": {"kind": "bag", "params": fb["tree_params"], "B": fb["n_estimators"], "seed": fb["seed"]},
        "boost_selected": {"kind": "boost", "n_rounds": fbo["n_rounds"], "max_depth": fbo["max_depth"],
                           "seed": fbo["seed"]},
    }
    res, res_masked, st = {}, {}, {}
    for name, spec in specs_a.items():
        Pc, Pm, stats = run_config(name, spec, D, resamples)
        res[name], res_masked[name], st[name] = bias_variance(Pc, y), bias_variance(Pm, y), mean_stats(stats)
    rows_a = [{"model": n, **res[n]} for n in specs_a]
    show(rows_a, ["model", "bias2", "variance", "expected_sq_error", "acc_mean", "acc_std"])

    # ---------------------------------------------------------------- (b) depth study
    print("\n=== Q3(b) depth study: single trees and 100-tree bagging at each depth ===")
    depth_cfgs = [  # (label, params, role)
        ("depth 4", {"max_depth": 4, "min_samples_leaf": msl}, "shallow"),
        ("depth 8", {"max_depth": 8, "min_samples_leaf": msl}, "extra"),
        ("depth 16", {"max_depth": 16, "min_samples_leaf": msl}, "extra"),
        ("selected", tree_params, "selected"),
        ("deepest", {"max_depth": None, "min_samples_leaf": 1}, "much deeper"),
    ]
    rows_b = []
    for label, params, role in depth_cfgs:
        key = label.replace(" ", "")
        if role == "selected":
            t_name = "tree_selected"
            b_name = "bag_selected" if fb["n_estimators"] == B_DEPTH_STUDY else "bag_depth_selected"
        else:
            t_name, b_name = f"tree_{key}", f"bag_{key}"
        for name, spec in ((t_name, {"kind": "tree", "params": params, "random_state": rs}),
                           (b_name, {"kind": "bag", "params": params, "B": B_DEPTH_STUDY, "seed": fb["seed"]})):
            if name not in res:
                Pc, Pm, stats = run_config(name, spec, D, resamples)
                res[name], res_masked[name], st[name] = bias_variance(Pc, y), bias_variance(Pm, y), mean_stats(stats)
        t, b = res[t_name], res[b_name]
        rows_b.append({
            "config": label, "role": role, "max_depth": str(params["max_depth"]),
            "min_leaf": params["min_samples_leaf"], "tree_depth": st[t_name]["depth"],
            "leaves": st[t_name]["n_leaves"], "samples_per_leaf": st[t_name]["mean_samples_per_leaf"],
            "single_face_leaves": st[t_name]["frac_single_face_leaves"],
            "tree_bias2": t["bias2"], "tree_var": t["variance"], "tree_acc": t["acc_mean"],
            "bag_bias2": b["bias2"], "bag_var": b["variance"], "bag_acc": b["acc_mean"],
            "var_removed": t["variance"] - b["variance"],
            "var_removed_frac": (t["variance"] - b["variance"]) / t["variance"] if t["variance"] > 0 else 0.0,
            "bias2_change": b["bias2"] - t["bias2"],
            "acc_gain": b["acc_mean"] - t["acc_mean"],
            "bag_disagreement": st[b_name]["disagreement_test"],
            "bag_tree_leaves": st[b_name]["n_leaves"],
            "bag_tree_single_face_leaves": st[b_name]["frac_single_face_leaves"],
        })
    show(rows_b, ["config", "role", "tree_depth", "leaves", "samples_per_leaf", "single_face_leaves"])
    print()
    show(rows_b, ["config", "tree_bias2", "tree_var", "bag_bias2", "bag_var", "var_removed", "var_removed_frac"])
    print()
    show(rows_b, ["config", "tree_acc", "bag_acc", "acc_gain", "bias2_change", "bag_disagreement",
                  "bag_tree_single_face_leaves"])

    # ---------------------------------------------------------------- (c) where does bagging help most
    best = max(rows_b, key=lambda r: r["acc_gain"])
    print(f"\n=== Q3(c) largest bagging accuracy gain: {best['config']} "
          f"(+{best['acc_gain']:.4f}); deepest: +{rows_b[-1]['acc_gain']:.4f} ===")

    # ---------------------------------------------------------------- (d) final table
    print("\n=== Q3(d) final table (accuracies of the frozen models from Q1/Q2) ===")
    d = q2["d_results"]
    rows_d = []
    for label, key, q2key in (("single tree", "tree_selected", "tree"), ("bagging", "bag_selected", "bagging"),
                              ("boosting", "boost_selected", "boosting")):
        rows_d.append({"model": label, "clean_test_acc": d[f"{q2key}_test"]["acc"],
                       "masked_test_acc": d[f"{q2key}_test_masked"]["acc"],
                       "bias2": res[key]["bias2"], "variance": res[key]["variance"],
                       "refit_acc_mean": res[key]["acc_mean"], "refit_acc_std": res[key]["acc_std"],
                       "masked_bias2": res_masked[key]["bias2"], "masked_variance": res_masked[key]["variance"]})
    show(rows_d, ["model", "clean_test_acc", "masked_test_acc", "bias2", "variance", "refit_acc_mean",
                  "refit_acc_std"])
    print()
    show(rows_d, ["model", "masked_bias2", "masked_variance"])
    tex = ["\\begin{tabular}{lcccc}", "\\toprule",
           "Model & Clean acc. & Masked acc. & Bias$^2$ & Variance \\\\", "\\midrule"]
    tex += [f"{r['model'].capitalize()} & {r['clean_test_acc']:.3f} & {r['masked_test_acc']:.3f} & "
            f"{r['bias2']:.3f} & {r['variance']:.3f} \\\\" for r in rows_d]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (C.RESULTS_DIR / "q3_final_table.tex").write_text("\n".join(tex) + "\n")

    # ---------------------------------------------------------------- figures
    lab = [f"{r['config']}\nNone, msl={r['min_leaf']}" if r["role"] in ("selected", "much deeper")
           else f"{r['config']}" + ("\n(shallow)" if r["role"] == "shallow" else "") for r in rows_b]
    x = np.arange(len(lab))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    for a, (tk, bk, title) in zip(ax, (("tree_bias2", "bag_bias2", "Bias²"), ("tree_var", "bag_var", "Variance"),
                                       ("tree_acc", "bag_acc", f"Mean test accuracy over {R} refits"))):
        a.bar(x - 0.2, [r[tk] for r in rows_b], 0.4, label="single tree")
        a.bar(x + 0.2, [r[bk] for r in rows_b], 0.4, label="bagging (100 trees)")
        a.set_xticks(x, lab, fontsize=7); a.set_title(title); a.legend(fontsize=8)
    for i, r in enumerate(rows_b):  # bagging accuracy gain above each pair
        ax[2].annotate(f"{r['acc_gain']:+.2f}", (i, max(r["tree_acc"], r["bag_acc"])),
                       textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    ax[2].set_ylim(0, max(max(r["bag_acc"], r["tree_acc"]) for r in rows_b) * 1.15)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q3b_depth_bias_variance.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    names = [r["model"] for r in rows_d]
    ax[0].bar(names, [r["bias2"] for r in rows_d], label="bias²")
    ax[0].bar(names, [r["variance"] for r in rows_d], bottom=[r["bias2"] for r in rows_d], label="variance")
    ax[0].set_title("Expected squared error = bias² + variance"); ax[0].legend(fontsize=8)
    xx = np.arange(len(names))
    ax[1].bar(xx - 0.2, [r["clean_test_acc"] for r in rows_d], 0.4, label="clean")
    ax[1].bar(xx + 0.2, [r["masked_test_acc"] for r in rows_d], 0.4, label="masked")
    ax[1].set_xticks(xx, names); ax[1].set_title("Test accuracy of the frozen models"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q3d_summary.png", dpi=200); plt.close(fig)

    out = {"R": R, "a": rows_a, "b": rows_b, "c_best_gain": best["config"], "d": rows_d,
           "all_configs": res, "all_configs_masked": res_masked, "leaf_and_disagreement_stats": st}
    (C.RESULTS_DIR / "q3.json").write_text(json.dumps(C.to_jsonable(out), indent=2))


if __name__ == "__main__":
    main()
