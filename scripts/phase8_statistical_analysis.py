"""
Phase 8 — Statistical analysis (no new experiments).
Produces results/metrics/phase8_statistical_analysis.json with:
  - Phase 5: effect size (Cohen's d), approximate paired t-test, CI
  - Phase 6: McNemar's test (paired hallucination), Wilson CIs, chi-square
"""
import json, math, os, sys
import numpy as np
from scipy import stats

ROOT = r"C:\Users\stream\Desktop\m2 pfe\MASTER2-A1"
sys.path.insert(0, ROOT)
os.chdir(ROOT)

# ---- Phase 5: fidelity (30 paired val tiles) -------------------------------
eval_d = json.load(open("results/metrics/phase5_edsr_small_eval.json"))
edsr_psnr = eval_d["edsr"]["metrics"]["psnr"]
bic_psnr = eval_d["bicubic"]["metrics"]["psnr"]
n5 = edsr_psnr["n"]
mean_diff = eval_d["delta_psnr_db"]
# pooled std of the difference (approx from per-metric stds)
std_edsr = edsr_psnr["std"]
std_bic = bic_psnr["std"]
pooled_std = math.sqrt((std_edsr**2 + std_bic**2) / 2)
cohens_d = mean_diff / pooled_std if pooled_std > 0 else float("inf")
# approximate paired t (needs per-pair values; we only have summary stats,
# so report the unpaired-style t as a conservative lower bound)
se = pooled_std * math.sqrt(2 / n5)
t_stat = mean_diff / se
p_val_approx = 2 * (1 - stats.t.cdf(abs(t_stat), df=n5 - 1))
# Wilson CI for the PSNR means
def wilson_mean_ci(mean, std, n, z=1.96):
    se = std / math.sqrt(n)
    return [mean - z * se, mean + z * se]

phase5_stats = {
    "n_pairs": n5,
    "edsr_psnr_mean": edsr_psnr["mean"],
    "edsr_psnr_std": std_edsr,
    "edsr_psnr_ci95": wilson_mean_ci(edsr_psnr["mean"], std_edsr, n5),
    "bicubic_psnr_mean": bic_psnr["mean"],
    "bicubic_psnr_std": std_bic,
    "bicubic_psnr_ci95": wilson_mean_ci(bic_psnr["mean"], std_bic, n5),
    "mean_diff_db": mean_diff,
    "cohens_d (approx)": round(cohens_d, 3),
    "t_stat (approx, unpaired lower bound)": round(t_stat, 3),
    "p_value (approx)": p_val_approx,
    "note": "Per-pair PSNR values were not saved in phase5_eval.json; "
            "t-test uses summary stats (conservative). For an exact paired "
            "t-test, re-run eval saving per-pair values.",
}

# ---- Phase 6: hallucination (1,742 paired tiles) --------------------------
recs = [json.loads(l) for l in open("results/metrics/phase6_per_tile.jsonl")]

def wilson_rate_ci(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / denom
    return [centre - half, centre + half]

def mcNemar(b, c):
    """McNemar's test for paired binary outcomes.
    b = EDSR hallucinated, bicubic not; c = bicubic hallucinated, EDSR not."""
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # continuity correction
    p = 1 - stats.chi2.cdf(chi2, df=1)
    return {"chi2": round(chi2, 4), "p_value": p, "b": b, "c": c}

cats = ["spectral", "structural", "textural"]
phase6_stats = {"n_tiles": len(recs)}
for cat in cats:
    e_det = [1 if r["edsr"][cat]["detected"] else 0 for r in recs]
    b_det = [1 if r["bicubic"][cat]["detected"] else 0 for r in recs]
    k_e, k_b = sum(e_det), sum(b_det)
    # discordant pairs
    b_disc = sum(1 for e, b in zip(e_det, b_det) if e == 1 and b == 0)
    c_disc = sum(1 for e, b in zip(e_det, b_det) if e == 0 and b == 1)
    phase6_stats[cat] = {
        "edsr_rate": k_e / len(recs),
        "edsr_n": k_e,
        "edsr_ci95": wilson_rate_ci(k_e, len(recs)),
        "bicubic_rate": k_b / len(recs),
        "bicubic_n": k_b,
        "bicubic_ci95": wilson_rate_ci(k_b, len(recs)),
        "rate_diff_pp": (k_e - k_b) / len(recs) * 100,
        "mcnemar": mcNemar(b_disc, c_disc),
    }

# any-hallucination (classification != 'none')
e_any = [1 if r["edsr"]["classification"] != "none" else 0 for r in recs]
b_any = [1 if r["bicubic"]["classification"] != "none" else 0 for r in recs]
b_disc = sum(1 for e, b in zip(e_any, b_any) if e == 1 and b == 0)
c_disc = sum(1 for e, b in zip(e_any, b_any) if e == 0 and b == 1)
phase6_stats["any_hallucination"] = {
    "edsr_rate": sum(e_any) / len(recs),
    "edsr_n": sum(e_any),
    "edsr_ci95": wilson_rate_ci(sum(e_any), len(recs)),
    "bicubic_rate": sum(b_any) / len(recs),
    "bicubic_n": sum(b_any),
    "bicubic_ci95": wilson_rate_ci(sum(b_any), len(recs)),
    "rate_diff_pp": (sum(e_any) - sum(b_any)) / len(recs) * 100,
    "mcnemar": mcNemar(b_disc, c_disc),
}

# chi-square test of independence (classification counts)
from scipy.stats import chi2_contingency
edsr_counts = {}
bic_counts = {}
for r in recs:
    edsr_counts[r["edsr"]["classification"]] = edsr_counts.get(r["edsr"]["classification"], 0) + 1
    bic_counts[r["bicubic"]["classification"]] = bic_counts.get(r["bicubic"]["classification"], 0) + 1
all_cats = sorted(set(list(edsr_counts.keys()) + list(bic_counts.keys())))
table = [[edsr_counts.get(c, 0) for c in all_cats],
         [bic_counts.get(c, 0) for c in all_cats]]
chi2, p_chi, dof, _ = chi2_contingency(table)
phase6_stats["chi_square_classification"] = {
    "chi2": round(chi2, 4),
    "p_value": p_chi,
    "dof": dof,
    "categories": all_cats,
    "edsr_counts": edsr_counts,
    "bicubic_counts": bic_counts,
}

# confidence distributions
for cat in cats:
    e_conf = [r["edsr"][cat].get("confidence", 0) for r in recs]
    b_conf = [r["bicubic"][cat].get("confidence", 0) for r in recs]
    phase6_stats[cat]["edsr_mean_confidence"] = float(np.mean(e_conf))
    phase6_stats[cat]["bicubic_mean_confidence"] = float(np.mean(b_conf))
    # Mann-Whitney U on confidences
    u, p_u = stats.mannwhitneyu(e_conf, b_conf, alternative="two-sided")
    phase6_stats[cat]["mannwhitney_confidence"] = {"U": round(u, 1), "p_value": p_u}

summary = {
    "phase": "8 - statistical analysis",
    "phase5": phase5_stats,
    "phase6": phase6_stats,
}
json.dump(summary, open("results/metrics/phase8_statistical_analysis.json", "w"), indent=2)
print(json.dumps(summary, indent=2)[:3000])
print("\nSaved -> results/metrics/phase8_statistical_analysis.json")
