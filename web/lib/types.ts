/**
 * Frontend mirror of `api/schemas.py`.
 *
 * Kept in sync manually for now — when api/schemas.py changes, update
 * this file too. (Long-term we could codegen from OpenAPI but it's
 * overkill for this project.)
 */

export interface Source {
  name: string;
  url: string;
}

export interface Treatment {
  display_name: string;
  is_healthy: boolean;
  is_background: boolean;
  severity?: "low" | "medium" | "high" | null;
  summary: string;
  time_sensitivity?: string | null;
  action_steps?: string[] | null;
  maintenance_tips?: string[] | null;
  retake_guidance?: string[] | null;
  sources: Source[];
}

export interface TopKPrediction {
  class_name: string;
  display_name: string;
  probability: number;
}

export interface PredictionResponse {
  class_name: string;
  display_name: string;
  confidence: number;
  is_healthy: boolean;
  is_background: boolean;
  top_k: TopKPrediction[];
  treatment: Treatment;
  warnings: string[];
  model_version: string;
  inference_time_ms: number;
}

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
  num_classes: number;
  model_version: string;
  tracking_uri: string;
  kb_version?: string | null;
  kb_entry_count?: number | null;
}

/**
 * Confidence bands per IMPLEMENTATION_GUIDE.md Decision 10.
 */
export type ConfidenceBand = "high" | "moderate" | "medium" | "low";

export function getConfidenceBand(c: number): ConfidenceBand {
  if (c >= 0.85) return "high";
  if (c >= 0.7) return "moderate";
  if (c >= 0.4) return "medium";
  return "low";
}

// ──────────────────────────────────────────────────────────────────────
// Training data (mirrors scripts/export_mlflow_for_dashboard.py output)
// ──────────────────────────────────────────────────────────────────────

export interface EpochMetric {
  epoch: number;
  epoch_in_phase: number;
  phase: "phase1" | "phase2";
  train_loss?: number;
  train_acc?: number;
  val_loss?: number;
  val_acc?: number;
  lr_head?: number;
  lr_backbone?: number;
  train_val_acc_gap?: number;
}

export interface PerClassMetric {
  index: number;
  class_name: string;
  display_name: string;
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface BestRun {
  attempt_id: string;
  parent_run_id: string;
  phase1_run_id: string | null;
  phase2_run_id: string | null;
  best_val_acc: number | null;
  best_epoch_global: number | null;
  test_acc: number | null;
  total_epochs: number;
  training_time_seconds: number | null;
  num_classes: number | null;
}

export interface TrainingData {
  _metadata: {
    generated_at: string;
    schema_version: number;
    source_attempt: string;
    tracking_uri: string;
    model_uri: string;
    data_root: string;
    seed: number;
  };
  best_run: BestRun;
  epoch_metrics: EpochMetric[];
  per_class_metrics: PerClassMetric[];
  confusion_matrix: number[][];
  class_names: string[];
}
