"""
Synthetic simulation of E2E test executions for the paper
"AI-Assisted Test Automation: Reducing Flaky Tests in End-to-End Frameworks".

ALL DATA IS SYNTHETIC. It is generated from assumed parameters to illustrate
the pipeline. It is NOT measured from any real system.
"""
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

import os
SEED = int(os.environ.get("SEED", 42))
rng = np.random.default_rng(SEED)

N_TESTS = 300
N_RUNS = 120            # CI runs
FLAKY_SHARE = 0.18      # share of tests that are intrinsically flaky
GENUINE_BUG_RUNS = 6    # number of "real defect" episodes injected

CATEGORIES = ["timing_async", "test_data_state", "order_dependency",
              "environment", "network_third_party", "locator_fragility", "concurrency"]
CAT_PROBS = [0.30, 0.20, 0.10, 0.12, 0.13, 0.10, 0.05]   # assumed mix
CAT_FAIL_RATE = {"timing_async": (0.04, 0.20), "test_data_state": (0.05, 0.25),
                 "order_dependency": (0.05, 0.30), "environment": (0.03, 0.15),
                 "network_third_party": (0.04, 0.22), "locator_fragility": (0.05, 0.25),
                 "concurrency": (0.04, 0.20)}

# ---- 1. Test population -------------------------------------------------
tests = pd.DataFrame({"test_id": range(N_TESTS)})
tests["is_flaky"] = rng.random(N_TESTS) < FLAKY_SHARE
tests["cause"] = None
fl = tests.is_flaky.values
tests.loc[fl, "cause"] = rng.choice(CATEGORIES, fl.sum(), p=CAT_PROBS)
tests["base_dur"] = rng.lognormal(mean=1.6, sigma=0.5, size=N_TESTS)   # seconds
tests["uses_css_xpath"] = rng.random(N_TESTS) < 0.35

def flaky_rate(cause):
    lo, hi = CAT_FAIL_RATE[cause]
    return rng.uniform(lo, hi)

tests["p_flake"] = [flaky_rate(c) if c else 0.0 for c in tests.cause]

# genuine defects: a set of tests fail consistently for a window of runs
bug_windows = []
for _ in range(GENUINE_BUG_RUNS):
    start = rng.integers(10, N_RUNS - 15)
    length = rng.integers(3, 12)
    affected = rng.choice(N_TESTS, rng.integers(2, 8), replace=False)
    bug_windows.append((start, start + length, set(affected.tolist())))

# ---- 2. Execution records ----------------------------------------------
rows = []
for run in range(N_RUNS):
    runner = rng.choice(["ci-a", "ci-b", "ci-c"], p=[0.5, 0.3, 0.2])
    runner_load = {"ci-a": 1.0, "ci-b": 1.15, "ci-c": 1.4}[runner]
    for t in tests.itertuples():
        genuine = any(s <= run < e and t.test_id in aff for s, e, aff in bug_windows)
        p = t.p_flake * (runner_load if t.cause == "environment" else 1.0)
        flaky_fail = (not genuine) and (rng.random() < p)
        failed = genuine or flaky_fail
        dur_noise = 0.12 + (0.35 if t.is_flaky else 0.0)
        dur = t.base_dur * rng.lognormal(0, dur_noise) * runner_load * (1.6 if flaky_fail else 1.0)
        # failure-level signals (noisy: genuine and flaky overlap)
        failed_requests = rng.poisson(1.8 if flaky_fail and t.cause == "network_third_party"
                                      else 0.6 if flaky_fail else 0.3)
        retry_passed = bool(failed and (flaky_fail or (rng.random() < 0.03)))
        err_type = "none"
        if failed:
            if flaky_fail:
                err_type = rng.choice(["timeout", "element_not_found", "assert_mismatch", "net_error"],
                                      p=[0.45, 0.25, 0.15, 0.15])
            else:
                err_type = rng.choice(["timeout", "element_not_found", "assert_mismatch", "net_error"],
                                      p=[0.15, 0.20, 0.55, 0.10])
        rows.append((run, t.test_id, runner, dur, int(failed), int(flaky_fail), int(genuine),
                     failed_requests, int(retry_passed), err_type, t.cause if flaky_fail else None))

df = pd.DataFrame(rows, columns=["run", "test_id", "runner", "duration", "failed", "flaky_fail",
                                 "genuine_fail", "failed_requests", "retry_passed", "err_type", "cause"])

# ---- 3. History features (only past runs: no leakage) -------------------
df = df.sort_values(["test_id", "run"]).reset_index(drop=True)
g = df.groupby("test_id")
df["prev_failed"] = g["failed"].shift(1)
df["flips"] = (df["failed"] != df["prev_failed"]).astype(int).where(df["prev_failed"].notna(), 0)
# cumulative stats up to previous run
df["hist_fail_rate"] = g["failed"].transform(lambda s: s.shift(1).expanding().mean()).fillna(0)
df["hist_flips"] = g["flips"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
df["hist_retry_pass"] = g["retry_passed"].transform(lambda s: s.shift(1).expanding().sum()).fillna(0)
df["dur_cv"] = g["duration"].transform(lambda s: (s.shift(1).expanding().std() /
                                                    s.shift(1).expanding().mean())).fillna(0)
df["dur_ratio"] = df["duration"] / g["duration"].transform(lambda s: s.shift(1).expanding().median()).fillna(df["duration"])

fails = df[df.failed == 1].copy()
fails = pd.concat([fails, pd.get_dummies(fails.err_type, prefix="err").astype(int),
                   pd.get_dummies(fails.runner, prefix="rn").astype(int)], axis=1)
feat = ["hist_fail_rate", "hist_flips", "hist_retry_pass", "dur_cv", "dur_ratio", "failed_requests"] + \
       [c for c in fails.columns if c.startswith("err_") and c != "err_type"] + \
       [c for c in fails.columns if c.startswith("rn_")]
fails["label"] = fails["flaky_fail"]   # 1 = flaky, 0 = genuine

# chronological split: first 70% of runs train, rest test
cut = int(N_RUNS * 0.7)
train, test = fails[fails.run < cut], fails[fails.run >= cut]
Xtr, ytr, Xte, yte = train[feat], train.label, test[feat], test.label

models = {
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
    "Gradient-boosted trees": GradientBoostingClassifier(random_state=SEED),
}
results = {}
for name, m in models.items():
    m.fit(Xtr, ytr)
    pr = m.predict(Xte)
    pb = m.predict_proba(Xte)[:, 1]
    results[name] = dict(precision=precision_score(yte, pr), recall=recall_score(yte, pr),
                         f1=f1_score(yte, pr), auc=roc_auc_score(yte, pb))

gb = models["Gradient-boosted trees"]
imp = sorted(zip(feat, gb.feature_importances_), key=lambda x: -x[1])[:6]

# Baseline: blind retry (a failure that passes on retry is treated as flaky; no classification)
blind = dict(precision=precision_score(yte, test.retry_passed),
             recall=recall_score(yte, test.retry_passed),
             f1=f1_score(yte, test.retry_passed))

# ---- 4. Dataset summary + root-cause distribution -----------------------
n_exec = len(df)
n_fail = int(df.failed.sum())
summary = dict(
    n_tests=N_TESTS, n_runs=N_RUNS, n_exec=n_exec, n_fail=n_fail,
    n_flaky_fail=int(df.flaky_fail.sum()), n_genuine_fail=int(df.genuine_fail.sum()),
    n_flaky_tests=int(tests.is_flaky.sum()),
    train_fail=len(train), test_fail=len(test),
    test_flaky=int(yte.sum()), test_genuine=int((1 - yte).sum()),
)
cause_counts = tests[tests.is_flaky].cause.value_counts()
cause_exec = df[df.flaky_fail == 1].cause.value_counts()
causes = {c: dict(tests=int(cause_counts.get(c, 0)),
                  failures=int(cause_exec.get(c, 0))) for c in CATEGORIES}

# ---- 5. Triage workload (classifier-filtered) ---------------------------
best = gb
pred_te = best.predict(Xte)
flagged_flaky = int(pred_te.sum())
missed_genuine = int(((pred_te == 1) & (yte == 0)).sum())      # genuine bugs mislabeled as flaky
triage = dict(total_test_failures=len(test), predicted_flaky=flagged_flaky,
              predicted_genuine=int(len(test) - flagged_flaky),
              genuine_mislabeled_flaky=missed_genuine,
              genuine_total=int((1 - yte).sum()))

# ---- 6. Scenario analysis (assumption-driven, NOT a finding) ------------
# Assume remediation removes a fixed share of flaky failures per cause.
ASSUMED_FIX = {"timing_async": 0.80, "test_data_state": 0.75, "order_dependency": 0.70,
               "environment": 0.30, "network_third_party": 0.40,
               "locator_fragility": 0.80, "concurrency": 0.50}
before_flaky_fail = int(df.flaky_fail.sum())
after_flaky_fail = sum(v * (1 - ASSUMED_FIX[c]) for c, v in cause_exec.items())
# pipelines (runs) with >=1 flaky failure, before vs expected after
runs_before = int((df.groupby("run").flaky_fail.sum() > 0).sum())
rs = np.random.default_rng(SEED + 1)
runs_after_list = []
for _ in range(200):
    cnt = np.zeros(N_RUNS, dtype=int)
    for r in df[df.flaky_fail == 1].itertuples():
        if rs.random() > ASSUMED_FIX[r.cause]:
            cnt[r.run] += 1
    runs_after_list.append(int((cnt > 0).sum()))
scenario = dict(flaky_fail_before=before_flaky_fail, flaky_fail_after=round(after_flaky_fail, 1),
                runs_with_flaky_before=runs_before,
                runs_with_flaky_after_mean=float(np.mean(runs_after_list)),
                runs_with_flaky_after_sd=float(np.std(runs_after_list)),
                assumed_fix=ASSUMED_FIX)

out = dict(summary=summary, causes=causes, models=results, blind_retry=blind,
           top_features=[(a, round(float(b), 3)) for a, b in imp], triage=triage, scenario=scenario)
print(json.dumps(out, indent=2))
json.dump(out, open(f"/home/claude/sim_results_seed{SEED}.json", "w"), indent=2)
