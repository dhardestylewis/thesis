import os
from pathlib import Path

# =====================================================================
# THESIS ARTIFACT REGISTRY
# =====================================================================
# This centralized module replaces hardcoded string pathing (os.path.join)
# across the 20+ pipeline scripts.
# 
# Usage:
#   from Analysis.Scripts.artifact_registry import TraceabilityRegistry
#   df = pd.read_csv(TraceabilityRegistry.STAGE_C_OOF_H0)
#
# Future Reorgs: 
#   If you decide to partition Track1_Predictive into /Models and /Metrics,
#   simply update the definitions below. All downstream visualization
#   and orchestration scripts will natively inherit the new structure.
# =====================================================================

ROOT_DIR = Path(r"C:\Users\dhl\data\thesis\thesis")

# Global Output Bounding Boxes
DATA_WAREHOUSE_DIR = ROOT_DIR / "Data" / "Warehouse_As_Of"
TRACK0_DIR = ROOT_DIR / "Analysis" / "Output" / "Track0_Predictive"
TRACK1_DIR = ROOT_DIR / "Analysis" / "Output" / "Track1_Predictive"
FIGURES_DIR = ROOT_DIR / "Thesis_Draft" / "Draft_v1" / "Figures"

class TraceabilityRegistry:
    """
    Registry isolating all structural artifacts for the thesis pipeline.
    Maintains compatibility with the current flat structure dynamically.
    """
    
    # Ensure directory existence upon load
    TRACK0_DIR.mkdir(parents=True, exist_ok=True)
    TRACK1_DIR.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # STAGE A: DEVELOPMENT HAZARD (Track0)
    # =========================================================
    STAGE_A_HAZARD_RESULTS = TRACK0_DIR / "stage_a_hazard_results.csv"
    STAGE_A_WINNER_H4 = TRACK0_DIR / "stage_a_winner_H=4.txt"
    
    # =========================================================
    # STAGE B: 6-TIER TYPOLOGY (Track1)
    # =========================================================
    STAGE_B_MODEL = TRACK1_DIR / "stage_b_model.cbm"
    STAGE_B_RESULTS = TRACK1_DIR / "StageB_Results.txt"
    
    # =========================================================
    # STAGE C: OPPOSITION RISK — COMPILED MODELS
    # =========================================================
    STAGE_C_MODEL_H0 = TRACK1_DIR / "stage_c_model_H0.joblib"
    STAGE_C_MODEL_H3 = TRACK1_DIR / "stage_c_model_H3.joblib"
    
    # =========================================================
    # STAGE C: OUT-OF-FOLD (OOF) PROBABILISTIC SEQUENCES
    # =========================================================
    STAGE_C_OOF_H0 = TRACK1_DIR / "stage_c_oof_predictions_H0.csv"
    STAGE_C_OOF_H3 = TRACK1_DIR / "stage_c_oof_predictions_H3.csv"
    STAGE_C_OOF_BASE = TRACK1_DIR / "stage_c_oof_predictions.csv"
    
    # =========================================================
    # STAGE C: MACRO EVALUATIONS (OOD, DRIFT, TOPOLOGIES)
    # =========================================================
    STAGE_C_DRIFT_H0 = TRACK1_DIR / "stage_c_drift_H0.csv"
    STAGE_C_DRIFT_H3 = TRACK1_DIR / "stage_c_drift_H3.csv"
    STAGE_C_REGIMES_H0 = TRACK1_DIR / "stage_c_regimes_H0.csv"
    STAGE_C_REGIMES_H3 = TRACK1_DIR / "stage_c_regimes_H3.csv"
    STAGE_C_FEATURE_IMPORTANCE_H0 = TRACK1_DIR / "stage_c_feature_importance_H0.csv"
    STAGE_C_FEATURE_IMPORTANCE_H3 = TRACK1_DIR / "stage_c_feature_importance_H3.csv"
    
    # =========================================================
    # STAGE D: INSTITUTIONAL OUTCOMES
    # =========================================================
    STAGE_D_RESULTS = TRACK1_DIR / "stage_d_results.txt"
    
    # =========================================================
    # STAGE E: NARRATIVE / AST TELEMETRY
    # =========================================================
    AST_STATE_JSON = TRACK1_DIR / "ast_state.json"
    MULTI_HORIZON_JSON = TRACK1_DIR / "multi_horizon_results.json"
    
    # =========================================================
    # STAGE F: GENERATIVE SIMULATION
    # =========================================================
    STAGE_F_SIMULATION = TRACK1_DIR / "stage_f_generative_simulation_results.csv"
    AUTOREGRESSIVE_IMPUTER = TRACK1_DIR / "stage_f_autoregressive_imputer_H0_to_H3.joblib"
    
    # =========================================================
    # SUMMARY / LEGACY
    # =========================================================
    TRACK1_RESULTS = TRACK1_DIR / "track1_results.csv"
    TRACK1_WAREHOUSE_EVAL = TRACK1_DIR / "Track1_Warehouse_Evaluation.csv"
    
    # ---------------------------------------------------------
    # Dynamic helpers for horizon-parameterized file names
    # ---------------------------------------------------------
    @staticmethod
    def stage_c_oof(safe_hz: str) -> Path:
        return TRACK1_DIR / f"stage_c_oof_predictions_{safe_hz}.csv"
    
    @staticmethod
    def stage_c_model(safe_hz: str) -> Path:
        return TRACK1_DIR / f"stage_c_model_{safe_hz}.joblib"
    
    @staticmethod
    def stage_c_drift(safe_hz: str) -> Path:
        return TRACK1_DIR / f"stage_c_drift_{safe_hz}.csv"
    
    @staticmethod
    def stage_c_regimes(safe_hz: str) -> Path:
        return TRACK1_DIR / f"stage_c_regimes_{safe_hz}.csv"

    @staticmethod
    def stage_c_feature_importance(safe_hz: str) -> Path:
        return TRACK1_DIR / f"stage_c_feature_importance_{safe_hz}.csv"

    @staticmethod
    def stage_a_model_lgbm(h_tag: str) -> Path:
        return TRACK0_DIR / f"stage_a_model_lgbm_{h_tag}.joblib"
    
    @staticmethod
    def stage_a_model_cb(h_tag: str) -> Path:
        return TRACK0_DIR / f"stage_a_model_cb_{h_tag}.cbm"
    
    @staticmethod
    def stage_a_winner(h_tag: str) -> Path:
        return TRACK0_DIR / f"stage_a_winner_{h_tag}.txt"
