"""Q2: SAMME AdaBoost from scratch.

Run:  python q2_boosting.py | tee results/q2_log.txt
"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import common as C

DEPTHS = [1, 2, 3, 4]
ROUNDS = [10, 25, 50, 100]
SIZES = [25, 50, 100]
BOOST_SEED_TAG = 3


def show(rows, cols):
    print(" | ".join(f"{c:>12}" for c in cols))
    for r in rows:
        print(" | ".join(f"{r[c]:>12.4f}" if isinstance(r[c], float) else f"{str(r[c]):>12}" for c in cols))


def load_frozen_q1(D):
    ft = json.loads((C.CONFIG_DIR / "frozen_tree.json").read_text())
    fb = json.loads((C.CONFIG_DIR / "frozen_bagging.json").read_text())
    tree = DecisionTreeClassifier(**ft).fit(D["X_train"], D["y_train"])
    bag = C.BaggedTrees(fb["n_estimators"], fb["tree_params"], fb["seed"]).fit(D["X_train"], D["y_train"])
    return ft, fb, tree, bag


# ----------------------------------------------------------------------------- (a) toy example
def part_a():
    print("\n=== Q2(a) toy 3-class example, one complete SAMME round ===")
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0], [9.0]])
    y = np.array([0, 0, 1, 0, 1, 1, 2, 1, 2])
    K = 3
    logs = []
    model = C.SAMME(n_rounds=1, max_depth=1, seed=C.child_seed(BOOST_SEED_TAG), n_classes=K)
    model.fit(X, y, round_log=logs.append)  # SAME class + update rule as the face experiments
    L = logs[0]
    lines = [f"n = {len(y)} samples, K = {K} classes, weak learner = depth-1 tree (stump)",
             f"stump: x <= {model.learners_[0].tree_.threshold[0]:.2f}",
             "  i   x  y  w_before  h(x)  miss  w*exp(alpha*miss)  w_after"]
    for i in range(len(y)):
        lines.append(f"{i:3d} {X[i, 0]:3.0f} {y[i]:2d}  {L['w_before'][i]:.4f}  {L['pred'][i]:4d}  "
                     f"{int(L['miss'][i]):4d}  {L['w_unnorm'][i]:17.4f}  {L['w_after'][i]:.4f}")
    eps, alpha = L["eps"], L["alpha"]
    Z = L["w_unnorm"].sum()
    lines += [f"weighted error eps = sum of w_before over misclassified = {eps:.4f}",
              f"alpha = log((1-eps)/eps) + log(K-1) = {np.log((1 - eps) / eps):.4f} + {np.log(K - 1):.4f} = {alpha:.4f}",
              f"misclassified weights multiplied by exp(alpha) = {np.exp(alpha):.4f}; normaliser Z = {Z:.4f}",
              f"after update: misclassified samples carry {L['w_after'][L['miss']].sum():.4f} of the weight "
              f"(was {eps:.4f}), correct samples carry {L['w_after'][~L['miss']].sum():.4f}",
              f"sum of w_after = {L['w_after'].sum():.4f}"]
    # independent hand check of the update rule
    a_chk = np.log((1 - eps) / eps) + np.log(K - 1)
    w_chk = L["w_before"] * np.exp(a_chk * L["miss"]); w_chk /= w_chk.sum()
    assert np.isclose(a_chk, alpha) and np.allclose(w_chk, L["w_after"])
    lines.append("hand check of alpha and weights: OK")
    text = "\n".join(lines)
    print(text)
    (C.RESULTS_DIR / "q2a_toy_round.txt").write_text(text + "\n")
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in L.items()}


# ----------------------------------------------------------------------------- (b) grid + plots
def part_b(D):
    print("\n=== Q2(b) SAMME on faces: depth x rounds (selection on clean validation only) ===")
    seed = C.child_seed(BOOST_SEED_TAG)
    rows, models = [], {}
    for d in DEPTHS:
        m = C.SAMME(max(ROUNDS), d, seed).fit(D["X_train"], D["y_train"])
        models[d] = m
        P = {s: m.learner_predictions(D[f"X_{s}"]) for s in ("train", "val")}
        for T in ROUNDS:
            Tu = min(T, len(m.learners_))
            r = {"depth": d, "rounds": T, "learners_used": Tu}
            for s in ("train", "val"):
                S = C.votes_from_predictions(P[s][:Tu], weights=m.alphas_[:Tu])
                r[f"{s}_acc"] = C.accuracy(D[f"y_{s}"], S.argmax(1))
            rows.append(r)
        print(f"depth {d}: {m.stop_reason_}")
    show(rows, ["depth", "rounds", "learners_used", "train_acc", "val_acc"])
    # rule: highest val acc; ties -> fewer rounds, then shallower learner
    best = max(range(len(rows)), key=lambda i: (rows[i]["val_acc"], -rows[i]["rounds"], -rows[i]["depth"]))
    sel = rows[best]
    frozen = {"max_depth": sel["depth"], "n_rounds": sel["rounds"], "seed": seed}
    (C.CONFIG_DIR / "frozen_boosting.json").write_text(json.dumps(frozen, indent=2))
    boost = C.SAMME(sel["rounds"], sel["depth"], seed).fit(D["X_train"], D["y_train"])
    sel = {**sel, "test_acc": C.accuracy(D["y_test"], boost.predict(D["X_test"])),
           "stop_reason": boost.stop_reason_}
    print("\nFROZEN boosting:", frozen, {k: v for k, v in sel.items() if k not in frozen})

    T = len(boost.learners_)
    t = np.arange(1, T + 1)
    W = boost.weight_history_[:T]  # weights used to train learner t
    top10 = np.sort(W, axis=1)[:, -10:].sum(1)
    n = W.shape[1]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    ax[0].plot(t, boost.errors_, "-o", ms=3); ax[0].axhline(1 - 1 / C.N_CLASSES, ls="--", c="r", lw=1,
                                                            label="random: 1-1/K")
    ax[0].set_title("Weighted training error ε_t"); ax[0].legend(fontsize=8)
    ax[1].plot(t, boost.alphas_, "-o", ms=3); ax[1].axhline(np.log(C.N_CLASSES - 1), ls=":", c="grey",
                                                            label="log(K-1)")
    ax[1].set_title("Learner weight α_t"); ax[1].legend(fontsize=8)
    ax[2].plot(t, W.max(1), "-o", ms=3); ax[2].axhline(1 / n, ls=":", c="grey", label="uniform 1/n")
    ax[2].set_yscale("log"); ax[2].set_title("Max sample weight"); ax[2].legend(fontsize=8)
    ax[3].plot(t, top10, "-o", ms=3); ax[3].axhline(10 / n, ls=":", c="grey", label="uniform 10/n")
    ax[3].set_title("Total weight of 10 heaviest images"); ax[3].legend(fontsize=8)
    for a in ax:
        a.set_xlabel("round t")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q2b_boosting_dynamics.png", dpi=200); plt.close(fig)
    dyn = {"eps": boost.errors_, "alpha": boost.alphas_, "max_w": W.max(1), "top10_w": top10}
    return rows, sel, frozen, boost, dyn


# ----------------------------------------------------------------------------- (c) vs bagging + heavy face
def part_c(D, fb, frozen, boost):
    print("\n=== Q2(c) boosting vs bagging with 25 / 50 / 100 learners ===")
    rows = []
    for M in SIZES:
        t0 = time.perf_counter()
        bag = C.BaggedTrees(M, fb["tree_params"], fb["seed"]).fit(D["X_train"], D["y_train"])
        t_bag = time.perf_counter() - t0
        t0 = time.perf_counter()
        bst = C.SAMME(M, frozen["max_depth"], frozen["seed"]).fit(D["X_train"], D["y_train"])
        t_bst = time.perf_counter() - t0
        for name, m, tt in (("bagging", bag, t_bag), ("boosting", bst, t_bst)):
            r = {"model": name, "learners": M, "fit_time_s": tt}
            if name == "boosting":
                r["learners"] = len(bst.learners_)
            for s in ("train", "val", "test"):
                r[f"{s}_acc"] = C.accuracy(D[f"y_{s}"], m.predict(D[f"X_{s}"]))
            rows.append(r)
    show(rows, ["model", "learners", "train_acc", "val_acc", "test_acc", "fit_time_s"])

    # heavy-weight training face in the frozen boosting model
    W = boost.weight_history_            # (T+1, n): W[t] = weights after t updates
    i = int(np.argmax(W[1:].max(0)))     # face reaching the largest weight at any round
    y = D["y_train"]
    hp = boost.train_pred_history_[:, i]  # weak learner predictions for face i
    T = len(hp)
    running = np.array([C.votes_from_predictions(boost.train_pred_history_[: t + 1, [i]],
                                                 weights=boost.alphas_[: t + 1]).argmax(1)[0]
                        for t in range(T)])
    wrong = hp[hp != y[i]]
    vals, cnts = np.unique(wrong, return_counts=True)
    common_wrong = int(vals[np.argmax(cnts)]) if len(vals) else None
    heavy = {"train_pos": i, "dataset_idx": int(D["idx_train"][i]), "identity": int(y[i]),
             "max_weight": float(W[1:, i].max()), "max_weight_round": int(np.argmax(W[1:, i]) + 1),
             "uniform_weight": 1 / len(y), "final_weight": float(W[-1, i]),
             "rank_final": int((W[-1] > W[-1, i]).sum() + 1),
             "learners_wrong": int((hp != y[i]).sum()), "learners_total": T,
             "most_common_wrong_class": common_wrong,
             "ensemble_correct_from_round": (int(np.argmax(running == y[i]) + 1)
                                             if (running == y[i]).any() else None),
             "ensemble_correct_at_end": bool(running[-1] == y[i]),
             "history": [{"t": t + 1, "w_used": float(W[t, i]), "h": int(hp[t]),
                          "h_correct": bool(hp[t] == y[i]), "ensemble_pred": int(running[t])}
                         for t in range(T)]}
    print("\nheavy-weight face:", {k: v for k, v in heavy.items() if k != "history"})
    print("round: weight used, weak-learner prediction (* = correct), running ensemble prediction")
    items = [f"{h['t']}:{h['w_used']:.4f},{h['h']}{'*' if h['h_correct'] else ''},{h['ensemble_pred']}"
             for h in heavy["history"]]
    for k in range(0, len(items), 8):
        print("  " + "  ".join(items[k:k + 8]))

    fig = plt.figure(figsize=(13, 3.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 3.2])
    a0 = fig.add_subplot(gs[0]); a0.imshow(D["img_train"][i], cmap="gray"); a0.axis("off")
    a0.set_title(f"heavy training face\nidentity {y[i]}", fontsize=9)
    a1 = fig.add_subplot(gs[1]); a1.axis("off")
    if common_wrong is not None:
        a1.imshow(D["img_train"][y == common_wrong][0], cmap="gray")
        a1.set_title(f"most frequent wrong\nlearner output: {common_wrong}", fontsize=9)
    a2 = fig.add_subplot(gs[2])
    ts = np.arange(1, T + 1)
    a2.plot(ts, W[:T, i], "-", c="grey", lw=1)
    ok = hp == y[i]
    a2.scatter(ts[ok], W[:T, i][ok], c="tab:green", s=18, label="weak learner correct", zorder=3)
    a2.scatter(ts[~ok], W[:T, i][~ok], c="tab:red", s=18, label="weak learner wrong", zorder=3)
    ens_ok = running == y[i]
    a2.scatter(ts, np.full(T, W[:T, i].min() * 0.5), c=np.where(ens_ok, "tab:green", "tab:red"),
               marker="s", s=10, label="ensemble correct / wrong (bottom row)")
    a2.axhline(1 / len(y), ls=":", c="k", lw=1, label="uniform 1/n")
    a2.set_yscale("log"); a2.set_xlabel("round t"); a2.set_ylabel("weight used at round t")
    a2.legend(fontsize=7, loc="upper left"); a2.set_title("Prediction history of the heavy face", fontsize=9)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "q2c_heavy_face.png", dpi=200); plt.close(fig)
    return rows, heavy


# ----------------------------------------------------------------------------- (d) clean vs masked
def mask_share(trees, weights=None):
    """(Weighted) share of split nodes that test a masked pixel. Mask = 256/4096 = 6.25% of pixels."""
    shares = np.array([np.isin(C.split_features(t), C.MASK_FEATURES).mean() if len(C.split_features(t))
                       else 0.0 for t in trees])
    w = np.ones(len(trees)) if weights is None else np.asarray(weights)
    return float((shares * w).sum() / w.sum())


def part_d(D, tree, bag, boost):
    print("\n=== Q2(d) single tree vs bagging vs boosting on clean and masked test ===")
    y = D["y_test"]
    out, cache = {}, {}
    for name in ("test", "test_masked"):
        X = D[f"X_{name}"]
        pt = tree.predict_proba(X); fbag = bag.vote_fractions(X); qb = boost.q(X)
        cache[name] = {"tree": pt, "bag": fbag, "boost": qb,
                       "bag_preds": bag.tree_predictions(X), "boost_preds": boost.learner_predictions(X)}
        for m, P in (("tree", pt), ("bagging", fbag), ("boosting", qb)):
            out[f"{m}_{name}"] = {"acc": C.accuracy(y, P.argmax(1)), "mean_win_conf": float(P.max(1).mean())}
    show([{"model": m, "clean_acc": out[f"{m}_test"]["acc"], "clean_conf": out[f"{m}_test"]["mean_win_conf"],
           "masked_acc": out[f"{m}_test_masked"]["acc"], "masked_conf": out[f"{m}_test_masked"]["mean_win_conf"]}
          for m in ("tree", "bagging", "boosting")],
         ["model", "clean_acc", "clean_conf", "masked_acc", "masked_conf"])
    shares = {"tree": mask_share([tree]), "bagging": mask_share(bag.trees_),
              "boosting_alpha_weighted": mask_share(boost.learners_, boost.alphas_),
              "pixel_share_of_mask": len(C.MASK_FEATURES) / 4096}
    print("share of split nodes testing masked pixels:", {k: round(v, 4) for k, v in shares.items()})

    # where the mask hits the single tree, and whether masked faces collapse onto few identities
    tt = tree.tree_
    depth_of = np.zeros(tt.node_count, int)
    for node in range(tt.node_count):  # sklearn numbers children after their parent
        for ch in (tt.children_left[node], tt.children_right[node]):
            if ch >= 0:
                depth_of[ch] = depth_of[node] + 1
    masked_nodes = np.flatnonzero(np.isin(tt.feature, C.MASK_FEATURES))
    tree_diag = {
        "masked_split_nodes": int(len(masked_nodes)),
        "masked_split_depths": sorted(depth_of[masked_nodes].tolist()),
        "masked_split_thresholds_positive": bool((tt.threshold[masked_nodes] > 0).all()),
        "frac_test_faces_path_uses_mask": float(np.mean([C.path_uses_mask(tree, x) for x in D["X_test"]])),
        "frac_tree_preds_changed_by_mask": float(np.mean(cache["test"]["tree"].argmax(1)
                                                         != cache["test_masked"]["tree"].argmax(1)))}
    print("single tree vs mask:", tree_diag)
    collapse = {}
    for mdl, key in (("tree", "tree"), ("bagging", "bag"), ("boosting", "boost")):
        for name in ("test", "test_masked"):
            pr = cache[name][key].argmax(1)
            collapse[f"{mdl}_{name}"] = {"distinct_predicted_ids": int(len(np.unique(pr))),
                                         "share_of_most_common_prediction": float(np.bincount(pr).max() / len(pr))}
    print("prediction spread (distinct identities predicted, share of the most common one):")
    for k, v in collapse.items():
        print(f"  {k}: {v}")
    shares = {**shares, "tree_diag": tree_diag, "collapse": collapse}

    # face where bagging and boosting respond differently to the mask
    c, m = cache["test"], cache["test_masked"]
    bag_ok = (c["bag"].argmax(1) == y) & (m["bag"].argmax(1) == y)
    bst_ok_clean = c["boost"].argmax(1) == y
    bst_ok_mask = m["boost"].argmax(1) == y
    bag_ok_clean = c["bag"].argmax(1) == y
    bag_ok_mask = m["bag"].argmax(1) == y
    groups = [("bagging holds, boosting breaks", np.flatnonzero(bag_ok & bst_ok_clean & ~bst_ok_mask)),
              ("boosting holds, bagging breaks", np.flatnonzero(bst_ok_clean & bst_ok_mask & bag_ok_clean & ~bag_ok_mask)),
              ("correctness differs on masked", np.flatnonzero(bag_ok_mask != bst_ok_mask))]
    kind, idx = next(((k, g) for k, g in groups if len(g)), (None, np.array([], int)))
    face = None
    if kind is not None:
        drop = (c["boost"][idx, y[idx]] - m["boost"][idx, y[idx]]) - (c["bag"][idx, y[idx]] - m["bag"][idx, y[idx]])
        j = int(idx[np.argmax(np.abs(drop))])
        B, a = len(bag.trees_), boost.alphas_

        def per_face_disagreement(frac):  # pairwise disagreement of trees on this one face
            return float((1 - (frac ** 2).sum()) * B / (B - 1))

        xc, xm = D["X_test"][j], D["X_test_masked"][j]
        bag_path = np.array([C.path_uses_mask(t, xc) for t in bag.trees_])
        bst_path = np.array([C.path_uses_mask(h, xc) for h in boost.learners_])
        flipped = c["boost_preds"][:, j] != m["boost_preds"][:, j]
        lost = (c["boost_preds"][:, j] == y[j]) & (m["boost_preds"][:, j] != y[j])
        Wf = boost.weight_history_[-1]
        same_id = D["y_train"] == y[j]
        face = {"kind": kind, "test_pos": j, "dataset_idx": int(D["idx_test"][j]), "true": int(y[j]),
                "bag_pred_clean": int(c["bag"][j].argmax()), "bag_pred_masked": int(m["bag"][j].argmax()),
                "bag_vote_true_clean": float(c["bag"][j, y[j]]), "bag_vote_true_masked": float(m["bag"][j, y[j]]),
                "bag_disagreement_clean": per_face_disagreement(c["bag"][j]),
                "bag_disagreement_masked": per_face_disagreement(m["bag"][j]),
                "bag_frac_trees_path_uses_mask": float(bag_path.mean()),
                "bag_frac_trees_changed_by_mask": float((c["bag_preds"][:, j] != m["bag_preds"][:, j]).mean()),
                "boost_pred_clean": int(c["boost"][j].argmax()), "boost_pred_masked": int(m["boost"][j].argmax()),
                "boost_q_true_clean": float(c["boost"][j, y[j]]), "boost_q_true_masked": float(m["boost"][j, y[j]]),
                "boost_alpha_share_path_uses_mask": float(a[bst_path].sum() / a.sum()),
                "boost_alpha_share_flipped": float(a[flipped].sum() / a.sum()),
                "boost_alpha_share_lost_true": float(a[lost].sum() / a.sum()),
                "boost_rounds_lost_true": (np.flatnonzero(lost) + 1).tolist(),
                "identity_final_weight_share": float(Wf[same_id].sum()),
                "identity_uniform_share": float(same_id.mean())}
        print("\nface with different mask response:")
        for k, v in face.items():
            print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")

        fig, ax = plt.subplots(1, 4, figsize=(14, 3.4))
        ax[0].imshow(D["img_test"][j], cmap="gray"); ax[0].set_title(f"clean, identity {y[j]}", fontsize=9)
        ax[1].imshow(D["img_test_masked"][j], cmap="gray"); ax[1].set_title("masked", fontsize=9)
        for a_ in ax[:2]:
            a_.axis("off")
        for k_, (key, title) in enumerate((("bag", "bagging vote fraction"), ("boost", "boosting q_k(x)"))):
            top = np.unique(np.concatenate([np.argsort(-c[key][j])[:4], np.argsort(-m[key][j])[:4], [y[j]]]))
            xx = np.arange(len(top))
            ax[2 + k_].bar(xx - 0.2, c[key][j, top], 0.4, label="clean")
            ax[2 + k_].bar(xx + 0.2, m[key][j, top], 0.4, label="masked")
            ax[2 + k_].set_xticks(xx, [f"{t}{'*' if t == y[j] else ''}" for t in top], fontsize=8)
            ax[2 + k_].set_title(f"{title} (* = true)", fontsize=9); ax[2 + k_].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(C.FIG_DIR / "q2d_mask_response.png", dpi=200); plt.close(fig)
    else:
        print("no face with different mask response found")
    return out, shares, face


def main():
    D = C.load_data()
    toy = part_a()
    ft, fb, tree, bag = load_frozen_q1(D)
    grid, sel, frozen, boost, dyn = part_b(D)
    comp, heavy = part_c(D, fb, frozen, boost)
    d_out, shares, face = part_d(D, tree, bag, boost)
    res = {"a_toy": toy, "b_grid": grid, "b_selected": sel, "b_frozen": frozen, "b_dynamics": dyn,
           "c_comparison": comp, "c_heavy_face": heavy, "d_results": d_out, "d_mask_split_share": shares,
           "d_face": face}
    (C.RESULTS_DIR / "q2.json").write_text(json.dumps(C.to_jsonable(res), indent=2))


if __name__ == "__main__":
    main()
