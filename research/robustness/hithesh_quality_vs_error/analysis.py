import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

def add_wrong(df):
    """Adds wrong flag and confidence column to the sweep dataframe."""
    df = df.copy()
    df["wrong"] = ((df["prob_pneumonia"] >= 0.5).astype(int) != df["label"]).astype(int)
    df["confidence"] = np.maximum(df["prob_pneumonia"], 1 - df["prob_pneumonia"])
    return df

def quality_error_auc(df, n_boot=500, seed=42):
    """Calculates error-detection AUC using score = 100 - quality, with a bootstrap CI."""
    df = add_wrong(df)
    y, s = df["wrong"].values, 100.0 - df["quality"].values
    
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan"), float("nan")
        
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) == 2:
            boots.append(roc_auc_score(y[idx], s[idx]))
            
    low, high = np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"), float("nan"))
    return float(auc), float(low), float(high)

def spearman_quality_confidence(df):
    """Calculates Spearman correlation between quality and model confidence."""
    df = add_wrong(df)
    rho, p_value = spearmanr(df["quality"], df["confidence"])
    return float(rho), float(p_value)

def rejection_curve(df, score_col="quality", coverages=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), seed=42):
    """Refuses lowest quality images first and compares against a random baseline."""
    df = add_wrong(df)
    wrong = df["wrong"].values
    n = len(wrong)
    
    # Sort descending by quality score (best quality first)
    order = np.argsort(-df[score_col].values)
    rng = np.random.default_rng(seed)
    rows = []
    
    for c in coverages:
        k = max(1, int(round(c * n)))
        by_quality = 1.0 - float(wrong[order[:k]].mean())
        by_random = float(np.mean([1.0 - wrong[rng.permutation(n)[:k]].mean() for _ in range(100)]))
        
        rows.append({
            "coverage": float(c),
            "accuracy_by_quality": round(by_quality, 4),
            "accuracy_random": round(by_random, 4)
        })
        
    return pd.DataFrame(rows)