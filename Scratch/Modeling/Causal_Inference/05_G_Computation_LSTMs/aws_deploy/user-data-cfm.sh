#!/bin/bash
# ==============================================================================
# user-data-cfm.sh
# Causal CFM CVAE training run on EC2 GPU instance.
# Logs to S3 every 30s; self-terminates on completion.
# ==============================================================================

exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
WORK_DIR="/home/ec2-user/causal_cfm_run"
OUT_DIR="$WORK_DIR/output"
PYTHON=""
for py in $(find /opt/conda/envs /opt/pytorch /home/ubuntu/anaconda3 -type f -name "python" 2>/dev/null); do
    if $py -c "import torch" 2>/dev/null; then
        PYTHON=$py
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "No pre-installed PyTorch found. Installing legacy cu118 PyTorch via system pip3..."
    if command -v yum >/dev/null 2>&1; then
        sudo yum update -y && sudo yum install -y python3-pip awscli
    else
        sudo apt-get update -y && sudo apt-get install -y python3-pip awscli
    fi
    PIP="python3 -m pip"
    PYTHON="python3"
    $PIP install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    $PIP install numpy pandas scikit-learn
else
    PIP=$(dirname $PYTHON)/pip
fi

echo "=== CAUSAL CFM CVAE TRAINING BOOTSTRAP ==="
echo "Python: $($PYTHON --version 2>&1)"

mkdir -p $WORK_DIR $OUT_DIR
cd $WORK_DIR

echo "[1/4] Pulling assets from S3..."
aws s3 cp ${S3_BUCKET}/src/causal_cfm_cvae.py .
aws s3 cp ${S3_BUCKET}/data/biweekly_panel.csv .
echo "  Assets pulled."

echo "[2/4] Installing remaining dependencies..."
$PIP install numpy scikit-learn pandas || true
echo "  Dependencies ready."

echo "[3/4] Starting S3 log sync daemon (every 30s)..."
(
  while true; do
    sleep 30
    aws s3 cp $OUT_DIR ${S3_BUCKET}/logs/ --recursive --exclude "*" --include "training_cfm_fold_*.log" > /dev/null 2>&1 || true
  done
) &
LOG_SYNC_PID=$!

echo "[4/4] Launching Causal CFM training (5 Folds Concurrently)..."
export OUT_DIR=$OUT_DIR
export PANEL_PATH=$WORK_DIR/biweekly_panel.csv
export S3_BUCKET=$S3_BUCKET

PIDS=""
for i in {0..4}; do
  echo "Starting Fold $i..."
  # Run PyTorch with --compile flag to fuse kernels!
  $PYTHON causal_cfm_cvae.py --fold $i --k_folds 5 --compile 2>&1 | tee $OUT_DIR/training_cfm_fold_${i}.log &
  PIDS="$PIDS $!"
done

echo "Waiting for all 5 folds to complete..."
wait $PIDS

echo "=== TRAINING COMPLETE ==="
kill $LOG_SYNC_PID 2>/dev/null || true

# Final uploads
aws s3 cp $OUT_DIR ${S3_BUCKET}/logs/ --recursive --exclude "*" --include "training_cfm_fold_*.log"
aws s3 cp /var/log/user-data.log        ${S3_BUCKET}/logs/cfm_user-data.log
aws s3 cp $OUT_DIR ${S3_BUCKET}/output/ --recursive --exclude "*" --include "causal_cfm_weights_fold_*.pt"
echo "CFM weights uploaded."

# Self-terminate
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
echo "Terminating instance $INSTANCE_ID in $REGION..."
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
