import json
import numpy as np
from src.models.matrix_factorization import MatrixFactorization

def save_model(model, dir_path):
    np.savez(
         f"{dir_path}/factors.npz",
        p = model._p, q = model._q,
        bias_u = model._bias_u, bias_i = model._bias_i,
    )

    meta = {
        "global_mean": float(model._global_mean),
        "user_idx": {str(k): v for k, v in model._user_idx.items()},
        "item_idx": {str(k): v for k, v in model._item_idx.items()},
        "n_factors": model.n_factors,
    }

    with open(f"{dir_path}/meta.json", "w") as f:
        json.dump(meta, f)

def load_model(dir_path):
    arrays = np.load(f"{dir_path}/factors.npz")
    with open(f"{dir_path}/meta.json") as f:
        meta = json.load(f)

    model = MatrixFactorization(n_factors = meta["n_factors"])
    model._p = arrays["p"]
    model._q = arrays["q"]
    model._bias_u = arrays["bias_u"]
    model._bias_i = arrays["bias_i"]
    model._global_mean = meta["global_mean"]
    model._user_idx = {int(k): v for k, v in meta["user_idx"].items()}
    model._item_idx = {int(k): v for k, v in meta["item_idx"].items()}
    model._idx_to_item = {idx: mid for mid, idx in model._item_idx.items()}
    
    return model

    