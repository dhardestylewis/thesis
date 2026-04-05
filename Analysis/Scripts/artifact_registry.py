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
    Maintains compatibility with the nested domain structure.
    """
    
    # ---------------------------------------------------------
    # DYNAMIC DIRECTORY RESOLUTION
    # ---------------------------------------------------------
    TRACK0_MODELS = TRACK0_DIR / "Models"
    TRACK0_METRICS = TRACK0_DIR / "Metrics"
    TRACK0_FIGURES = TRACK0_DIR / "Figures"

    TRACK1_MODELS = TRACK1_DIR / "Models"
    TRACK1_METRICS = TRACK1_DIR / "Metrics"
    TRACK1_TELEMETRY = TRACK1_DIR / "Telemetry"
    TRACK1_SIMULATION = TRACK1_DIR / "Simulation_Outputs"
    
    # Track 2 & 3 (Causal Models)
    TRACK2_DIR = ROOT_DIR / "Analysis" / "Output" / "Track2_Causal"
    TRACK2_METRICS = TRACK2_DIR / "Metrics"
    
    TRACK3_DIR = ROOT_DIR / "Analysis" / "Output" / "Track3_Causal"
    TRACK3_METRICS = TRACK3_DIR / "Metrics"
    
    # Analytical Pipelines
    ECONOMETRICS_DIR = ROOT_DIR / "Analysis" / "Output" / "Econometrics"
    ECONOMETRICS_METRICS = ECONOMETRICS_DIR / "Metrics"
    ECONOMETRICS_FIGURES = ECONOMETRICS_DIR / "Figures"
    
    FORECASTING_DIR = ROOT_DIR / "Analysis" / "Output" / "Forecasting"
    FORECASTING_METRICS = FORECASTING_DIR / "Metrics"
    FORECASTING_FIGURES = FORECASTING_DIR / "Figures"
    
    DESCRIPTIVE_DIR = ROOT_DIR / "Analysis" / "Output" / "Descriptive"
    DESCRIPTIVE_TABLES = DESCRIPTIVE_DIR / "Tables"
    DESCRIPTIVE_FIGURES = DESCRIPTIVE_DIR / "Figures"
    DESCRIPTIVE_LOGS = DESCRIPTIVE_DIR / "Logs"
    
    # Ensure directory existence upon load safely
    _subdirs = [
        TRACK0_MODELS, TRACK0_METRICS, TRACK0_FIGURES, 
        TRACK1_MODELS, TRACK1_METRICS, TRACK1_TELEMETRY, TRACK1_SIMULATION,
        TRACK2_METRICS, TRACK3_METRICS,
        ECONOMETRICS_METRICS, ECONOMETRICS_FIGURES,
        FORECASTING_METRICS, FORECASTING_FIGURES,
        DESCRIPTIVE_TABLES, DESCRIPTIVE_FIGURES, DESCRIPTIVE_LOGS
    ]
    for _p in _subdirs:
        _p.mkdir(parents=True, exist_ok=True)
    
    # =========================================================
    # STAGE A: DEVELOPMENT HAZARD (Track0)
    # =========================================================
    STAGE_A_HAZARD_RESULTS = TRACK0_DIR / "stage_a_hazard_results.csv" # Root matrix
    STAGE_A_WINNER_H4 = TRACK0_METRICS / "stage_a_winner_H=4.txt"
    
    # =========================================================
    # STAGE B: 6-TIER TYPOLOGY (Track1)
    # =========================================================
    STAGE_B_MODEL = TRACK1_MODELS / "stage_b_model.cbm"
    STAGE_B_RESULTS = TRACK1_METRICS / "StageB_Results.txt"
    
    # =========================================================
    # STAGE C: OPPOSITION RISK — COMPILED MODELS
    # =========================================================
    STAGE_C_MODEL_H0 = TRACK1_MODELS / "stage_c_model_H0.joblib"
    STAGE_C_MODEL_H3 = TRACK1_MODELS / "stage_c_model_H3.joblib"
    
    # =========================================================
    # STAGE C: OUT-OF-FOLD (OOF) PROBABILISTIC SEQUENCES
    # =========================================================
    STAGE_C_OOF_H0 = TRACK1_METRICS / "stage_c_oof_predictions_H0.csv"
    STAGE_C_OOF_H3 = TRACK1_METRICS / "stage_c_oof_predictions_H3.csv"
    STAGE_C_OOF_BASE = TRACK1_METRICS / "stage_c_oof_predictions.csv"
    
    # =========================================================
    # STAGE C: MACRO EVALUATIONS (OOD, DRIFT, TOPOLOGIES)
    # =========================================================
    STAGE_C_DRIFT_H0 = TRACK1_METRICS / "stage_c_drift_H0.csv"
    STAGE_C_DRIFT_H3 = TRACK1_METRICS / "stage_c_drift_H3.csv"
    STAGE_C_REGIMES_H0 = TRACK1_METRICS / "stage_c_regimes_H0.csv"
    STAGE_C_REGIMES_H3 = TRACK1_METRICS / "stage_c_regimes_H3.csv"
    STAGE_C_FEATURE_IMPORTANCE_H0 = TRACK1_METRICS / "stage_c_feature_importance_H0.csv"
    STAGE_C_FEATURE_IMPORTANCE_H3 = TRACK1_METRICS / "stage_c_feature_importance_H3.csv"
    
    # =========================================================
    # STAGE D: INSTITUTIONAL OUTCOMES
    # =========================================================
    STAGE_D_RESULTS = TRACK1_METRICS / "stage_d_results.txt"
    
    # =========================================================
    # STAGE E: NARRATIVE / AST TELEMETRY
    # =========================================================
    AST_STATE_JSON = TRACK1_TELEMETRY / "ast_state.json"
    MULTI_HORIZON_JSON = TRACK1_TELEMETRY / "multi_horizon_results.json"
    
    # =========================================================
    # STAGE F: GENERATIVE SIMULATION
    # =========================================================
    STAGE_F_SIMULATION = TRACK1_SIMULATION / "stage_f_generative_simulation_results.csv"
    AUTOREGRESSIVE_IMPUTER = TRACK1_MODELS / "stage_f_autoregressive_imputer_H0_to_H3.joblib"
    
    # =========================================================
    # SUMMARY / LEGACY
    # =========================================================
    TRACK1_RESULTS = TRACK1_METRICS / "track1_results.csv"
    TRACK1_WAREHOUSE_EVAL = TRACK1_METRICS / "Track1_Warehouse_Evaluation.csv"
    
    # ---------------------------------------------------------
    # Dynamic helpers for horizon-parameterized file names
    # ---------------------------------------------------------
    @staticmethod
    def stage_c_oof(safe_hz: str) -> Path:
        return TraceabilityRegistry.TRACK1_METRICS / f"stage_c_oof_predictions_{safe_hz}.csv"
    
    @staticmethod
    def stage_c_model(safe_hz: str) -> Path:
        return TraceabilityRegistry.TRACK1_MODELS / f"stage_c_model_{safe_hz}.joblib"
    
    @staticmethod
    def stage_c_drift(safe_hz: str) -> Path:
        return TraceabilityRegistry.TRACK1_METRICS / f"stage_c_drift_{safe_hz}.csv"
    
    @staticmethod
    def stage_c_regimes(safe_hz: str) -> Path:
        return TraceabilityRegistry.TRACK1_METRICS / f"stage_c_regimes_{safe_hz}.csv"

    @staticmethod
    def stage_c_feature_importance(safe_hz: str) -> Path:
        return TraceabilityRegistry.TRACK1_METRICS / f"stage_c_feature_importance_{safe_hz}.csv"

    @staticmethod
    def stage_a_model_lgbm(h_tag: str) -> Path:
        return TraceabilityRegistry.TRACK0_MODELS / f"stage_a_model_lgbm_{h_tag}.joblib"
    
    @staticmethod
    def stage_a_model_cb(h_tag: str) -> Path:
        return TraceabilityRegistry.TRACK0_MODELS / f"stage_a_model_cb_{h_tag}.cbm"
    
    @staticmethod
    def stage_a_winner(h_tag: str) -> Path:
        return TraceabilityRegistry.TRACK0_METRICS / f"stage_a_winner_{h_tag}.txt"
