#!/bin/bash
# ==============================================================================
# user-data.sh v5
# Uses absolute conda paths — no conda activation needed, no set -e
# Logs to /var/log/user-data.log (visible in EC2 console) + S3 every 30s
# ==============================================================================

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
WORK_DIR="/home/ec2-user/causal_run"
OUT_DIR="$WORK_DIR/output"
PYTHON="/opt/pytorch/bin/python"
PIP="/opt/pytorch/bin/pip"

echo "=== [v5] STARTING CAUSAL PIPELINE BOOTSTRAP ==="
echo "Python: $($PYTHON --version 2>&1)"
echo "Pip: $($PIP --version 2>&1)"

mkdir -p $WORK_DIR $OUT_DIR
cd $WORK_DIR

echo "[1/4] Pulling assets from S3..."
aws s3 cp ${S3_BUCKET}/src/causal_cfm_cvae.py .
aws s3 cp ${S3_BUCKET}/data/biweekly_panel.csv .
echo "  Assets pulled."

echo "[2/4] Installing scikit-learn..."
$PIP install scikit-learn --quiet
echo "  scikit-learn ready."

echo "[3/4] Starting S3 log sync daemon (every 30s)..."
(
  while true; do
    sleep 30
    if [ -f "$OUT_DIR/training_pipeline.log" ]; then
      aws s3 cp $OUT_DIR/training_pipeline.log ${S3_BUCKET}/logs/training_pipeline.log > /dev/null 2>&1
    fi
  done
) &
LOG_SYNC_PID=$!

echo "[4/4] Launching causal pipeline..."
for FOLD in {0..4}; do
    echo "Starting Fold $FOLD..." | tee -a $OUT_DIR/training_pipeline.log
    OUT_DIR=$OUT_DIR \
    PANEL_PATH=$WORK_DIR/biweekly_panel.csv \
      $PYTHON causal_cfm_cvae.py --fold $FOLD 2>&1 | tee -a $OUT_DIR/training_pipeline.log
done

echo "=== PIPELINE COMPLETE ==="
kill $LOG_SYNC_PID 2>/dev/null || true

# Final uploads
aws s3 cp $OUT_DIR/training_pipeline.log ${S3_BUCKET}/logs/training_pipeline.log
aws s3 cp /var/log/user-data.log ${S3_BUCKET}/logs/user-data.log
if [ -f "$OUT_DIR/vae_dose_response_surface_expanded.csv" ]; then
  aws s3 cp $OUT_DIR/vae_dose_response_surface_expanded.csv ${S3_BUCKET}/output/vae_dose_response_surface_expanded.csv
  echo "God Table uploaded to S3."
fi

# Upload weights
if [ -f "$OUT_DIR/causal_cfm_weights.pt" ]; then
  aws s3 cp $OUT_DIR/causal_cfm_weights.pt ${S3_BUCKET}/output/causal_cfm_weights.pt
  echo "Final Model weights uploaded to S3."
fi

# Upload all 5 Fold Checkpoints
for i in {0..4}; do
  if [ -f "$OUT_DIR/causal_cfm_weights_fold_${i}.pt" ]; then
    aws s3 cp $OUT_DIR/causal_cfm_weights_fold_${i}.pt ${S3_BUCKET}/output/causal_cfm_weights_fold_${i}.pt
    echo "Fold ${i} weights uploaded to S3."
  fi
done

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
echo "Terminating instance $INSTANCE_ID..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
