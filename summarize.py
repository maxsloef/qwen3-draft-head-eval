"""Render text tables from a results dict (pure stdlib). Runs inside the Modal job."""


def render(r):
    lines = []
    print = lambda *a: lines.append(" ".join(str(x) for x in a))
    M, K, cats = r["metrics"], r["K"], r["cats"]
    ix = {m: i for i, m in enumerate(M)}

    def tot(stats, cat=None):
        rows = [stats[c] for c in range(len(cats)) if cat is None or cats[c] == cat]
        return [[sum(x[s][m] for x in rows) for m in range(len(M))] for s in range(K)]

    for cond, c in r["conds"].items():

        print(f"\n=== {cond} ===")
        t0 = tot(next(iter(c["stats"].values())))
        print(f"anchors: {int(t0[0][0])}; trunk re-forward agrees with its own generated token: "
              + " ".join(f"{t0[s][ix['base_top1']] / t0[s][0]:.3f}" for s in range(K)))
        for metric, label in [("top1", "top-1 accuracy at step s (earlier draft tokens correct)"), ("top5", "top-5 accuracy"),
                              ("overlap", "distribution overlap with trunk (1 - TV)"), ("prefix", "all of steps 1..s correct"),
                              ("covered", "true token is in the head's 32K vocabulary")]:
            print(f"\n{label}\n{'head':10s} " + " ".join(f"s={s + 1:<4d}" for s in range(K)) + ("  mean accepted" if metric == "prefix" else ""))
            for n, st in c["stats"].items():
                t = tot(st)
                vals = [t[s][ix[metric]] / t[s][0] for s in range(K)]
                print(f"{n:10s} " + " ".join(f"{v:.3f} " for v in vals) + (f"  {sum(vals):.2f}" if metric == "prefix" else ""))
        print("\nmean accepted draft tokens (chain of 7) by category")
        print(f"{'head':10s} " + " ".join(f"{k[:9]:>9s}" for k in cats))
        for n, st in c["stats"].items():
            print(f"{n:10s} " + " ".join(f"{sum(tot(st, k)[s][ix['prefix']] / max(tot(st, k)[s][0], 1) for s in range(K)):9.2f}" for k in cats))
        print("\nconvention check, step-1 top-1 by (norm_before_residual, layer offset)")
        print(f"{'head':10s} " + " ".join(f"{str(tuple(v)):>12s}" for v in r["variants"]))
        for n, v in c["variants"].items():
            print(f"{n:10s} " + " ".join(f"{x[1] / max(x[0], 1):12.3f}" for x in v))
    return "\n".join(lines)
