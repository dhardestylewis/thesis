import os, json, torch, numpy as np, pandas as pd
import plotly.graph_objects as go

import causal_cfm_cvae
from causal_cfm_cvae import load_data, CausalSeq2SeqCFM, HEIGHT_IDX

BASE_DIR = os.environ.get("BASE_DIR", ".")
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(BASE_DIR, "output"))
PANEL_PATH = os.environ.get("PANEL_PATH", os.path.join(BASE_DIR, "biweekly_panel.csv"))
HTML_OUT_DIR = os.environ.get("HTML_OUT_DIR", OUT_DIR)

causal_cfm_cvae.PANEL_PATH = PANEL_PATH
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_manifest(fold: int) -> dict:
    path = os.path.join(OUT_DIR, f"feature_manifest_fold_{fold}.json")
    with open(path) as f:
        return json.load(f)


def build_model_from_manifest(m: dict, treat_idx: list[int]) -> CausalSeq2SeqCFM:
    return CausalSeq2SeqCFM(
        input_dim=m["input_dim"],
        y_dim=m["y_dim"],
        hidden_dim=m.get("hidden_dim", 256),
        latent_dim=m.get("latent_dim", 64),
        cfm_hidden=m.get("cfm_hidden", 512),
        cfm_layers=m.get("cfm_layers", 5),
        n_layers=m.get("n_layers", 3),
        treat_idx=treat_idx,
        skip_confounder_idx=m.get("skip_confounder_idx", []),
        prop_petition_idx=m.get("prop_petition_idx", []),
        treatment_derived_idx=m.get("treatment_derived_idx", []),
    ).to(device)


def load_fold(fold: int, features: list[str], targets: list[str], treat_idx: list[int]):
    m = load_manifest(fold)
    if features != m["feature_names"]:
        raise ValueError(f"Feature mismatch for fold {fold}. Refusing to run stale inference.")
    if targets != m["target_names"]:
        raise ValueError(f"Target mismatch for fold {fold}. Refusing to run stale inference.")

    model = build_model_from_manifest(m, treat_idx)
    ckpt = os.path.join(OUT_DIR, f"causal_cfm_weights_fold_{fold}.pt")
    sd = torch.load(ckpt, map_location=device)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch fold={fold}: missing={missing[:5]}, unexpected={unexpected[:5]}")
    model.eval()
    return model


def summarize_overlap(X: torch.Tensor, treat_idx: list[int], dose_grid: np.ndarray):
    obs = X[:, 4:, treat_idx[0]].detach().cpu().numpy().reshape(-1)
    obs = obs[np.isfinite(obs)]
    q = np.quantile(obs, [0, .25, .5, .75, .9, .95, .99, 1])
    print("Observed petition dose quantiles:", q)
    for d in dose_grid:
        print(f"dose={d:.2f} | within observed support: {q[0] <= d <= q[-1]}")


def generate_surfaces():
    X, Y, L, features, targets, norm_dict, treat_idx, cases, cell_assignments, filing_years = load_data()
    X = X.to(device)

    folds = [0, 1, 2, 3, 4]
    models = [load_fold(f, features, targets, treat_idx) for f in folds]

    height_idx = targets.index("height_concession_pct")
    dose_grid = np.linspace(0.0, 1.0, 11)
    summarize_overlap(X, treat_idx, dose_grid)

    # Example: overall surface by filing year bucket. Replace cohorts as needed.
    cohorts = {
        "pre_2020": np.where(filing_years.loc[cases].values < 2020)[0],
        "2020_2022": np.where((filing_years.loc[cases].values >= 2020) & (filing_years.loc[cases].values <= 2022))[0],
        "post_2022": np.where(filing_years.loc[cases].values > 2022)[0],
    }

    Z_mean = np.full((len(cohorts), len(dose_grid)), np.nan)
    Z_q10 = np.full_like(Z_mean, np.nan)
    Z_q90 = np.full_like(Z_mean, np.nan)

    for ci, (label, idx) in enumerate(cohorts.items()):
        if len(idx) == 0:
            continue
        idx = idx[:300]
        X_base = X[idx].clone()
        B, T, _ = X_base.shape

        for di, d in enumerate(dose_grid):
            X_cf = X_base.clone()

            # Counterfactual dose path in raw fraction units.
            dose_seq = torch.full((B, T - 4), float(d), device=device)

            fold_preds = []
            with torch.no_grad():
                for m in models:
                    preds = m.sample(X_cf[:, :4, :], X_cf.clone(), dose_val=dose_seq, n_steps=20)
                    fold_preds.append(preds[:, -1, height_idx].detach().cpu().numpy())

            terminal_height = np.mean(np.stack(fold_preds, axis=0), axis=0)

            # For zero-inflated height, mean expected height is the estimand;
            # p50 is expected to collapse toward zero and should not drive ATE surfaces.
            Z_mean[ci, di] = terminal_height.mean()
            Z_q10[ci, di] = np.quantile(terminal_height, 0.10)
            Z_q90[ci, di] = np.quantile(terminal_height, 0.90)

        print(label, "dose=0:", Z_mean[ci, 0], "dose=1:", Z_mean[ci, -1])

    fig = go.Figure(data=[
        go.Surface(
            z=Z_mean.T,
            x=np.arange(len(cohorts)),
            y=dose_grid * 100,
            colorbar=dict(title="E[height concession pct]"),
            name="Mean expected height",
        )
    ])
    fig.update_layout(
        title="Zero-inflated hurdle CFM: mean expected height surface",
        scene=dict(
            xaxis=dict(title="Cohort", tickvals=list(range(len(cohorts))), ticktext=list(cohorts.keys())),
            yaxis_title="Counterfactual petition dose (%)",
            zaxis_title="Mean expected height concession pct",
        ),
        width=1200,
        height=900,
    )
    out_path = os.path.join(HTML_OUT_DIR, "hurdle_cfm_height_mean_surface.html")
    fig.write_html(out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    generate_surfaces()
