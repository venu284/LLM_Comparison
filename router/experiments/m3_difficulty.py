"""A5: predict task difficulty b_q from the prompt, on the tasks whose prompts could be joined.

Run from the repo root:  pipeline/venv/bin/python -m router.experiments.m3_difficulty
Target b_q comes from a 1PL fit on all 9,999 ID tasks x 8 models (binarised at 0.5).
Evaluation: 5-fold cross-validation over joined tasks, grouped by prompt text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from router.core import irt
from router.data import coderouterbench as crb
from router.data.prompts import join
from router.data.scoring import ACCEPT_THRESHOLD
from router.settings import REPO_ROOT

OUT = REPO_ROOT / "router" / "experiments" / "out"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def main() -> None:
    probe, test = crb.load("probing"), crb.load("id_test")
    score = pd.concat([probe.score, test.score])
    dim = pd.concat([probe.dimension, test.dimension])
    src = pd.concat([probe.source, test.source])
    Y = (score.T.to_numpy() >= ACCEPT_THRESHOLD).astype(float)
    params = irt.fit(Y, score.index, score.columns, "1pl")
    b = pd.Series(params.b, index=score.index)

    prompts = join(score.index).dropna()
    tasks = prompts.index
    print("joined tasks:", len(tasks), "dimensions:", dim.loc[tasks].value_counts().to_dict())

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer(EMBED_MODEL, device="cpu")
    E = enc.encode([p[:2000] for p in prompts.tolist()], batch_size=64, normalize_embeddings=True,
                   show_progress_bar=False)

    onehot = pd.get_dummies(pd.DataFrame({"d": dim.loc[tasks], "s": src.loc[tasks]})).to_numpy(float)
    yb = b.loc[tasks].to_numpy()
    preds = {k: np.zeros(len(tasks)) for k in ("dimension mean", "source mean", "MiniLM + ridge",
                                               "MiniLM + source one-hot + ridge")}
    # Variants of one problem (e.g. HumanEval, its completion and test-generation versions, and
    # the HumanEval-X languages) share prompt text. Group folds by prompt so no problem appears
    # in both the training and the test fold.
    groups = pd.factorize(prompts.to_numpy())[0]
    print("distinct prompts:", groups.max() + 1)
    for tr, te in GroupKFold(5).split(tasks, groups=groups):
        t_tr = tasks[tr]
        dm = b.loc[t_tr].groupby(dim.loc[t_tr]).mean()
        sm = b.loc[t_tr].groupby(src.loc[t_tr]).mean()
        preds["dimension mean"][te] = dim.loc[tasks[te]].map(dm).fillna(b.loc[t_tr].mean()).to_numpy()
        preds["source mean"][te] = src.loc[tasks[te]].map(sm).fillna(b.loc[t_tr].mean()).to_numpy()
        r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(E[tr], yb[tr])
        preds["MiniLM + ridge"][te] = r.predict(E[te])
        X = np.hstack([E, onehot])
        r = RidgeCV(alphas=np.logspace(-2, 3, 12)).fit(X[tr], yb[tr])
        preds["MiniLM + source one-hot + ridge"][te] = r.predict(X[te])

    # Routing relevance: AUC of sigma(theta_m - b_hat) against the real 0/1 outcomes.
    th = pd.Series(params.theta, index=score.columns)
    Ysub = (score.loc[tasks] >= ACCEPT_THRESHOLD).astype(float)
    rows = []
    for name, p in preds.items():
        P = expit(th.to_numpy()[None, :] - p[:, None])
        rows.append({"predictor": name,
                     "spearman_b": spearmanr(yb, p).statistic,
                     "mae_b": float(np.abs(yb - p).mean()),
                     "pass_auc": roc_auc_score(Ysub.to_numpy().ravel(), P.ravel())})
    P = expit(th.to_numpy()[None, :] - yb[:, None])
    rows.append({"predictor": "fitted b (in-sample ceiling)", "spearman_b": 1.0, "mae_b": 0.0,
                 "pass_auc": roc_auc_score(Ysub.to_numpy().ravel(), P.ravel())})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "a5_difficulty.csv", index=False)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
