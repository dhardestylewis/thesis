#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting Causal CFM-CVAE Pipeline Setup ==="

S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
WORK_DIR="/home/ubuntu/causal_run"
OUT_DIR="$WORK_DIR/output"

mkdir -p $WORK_DIR $OUT_DIR
cd $WORK_DIR

# Install dependencies
echo "Installing base dependencies..."
apt-get update -y
apt-get install -y awscli python3-pip

# Pull assets from S3
echo "Downloading assets from S3..."
aws s3 cp ${S3_BUCKET}/src/ . --recursive
aws s3 cp ${S3_BUCKET}/data/biweekly_panel.csv biweekly_panel.csv

# Install packages directly — AMI already has CUDA/cuDNN, no Docker needed
echo "Installing required packages via pip..."
pip3 install --upgrade pip "setuptools<70" wheel --quiet
pip3 install pandas numpy scikit-learn pyarrow --quiet
pip3 install torch --index-url https://download.pytorch.org/whl/cu118 --quiet
pip3 install "catboost>=1.2,<1.3" "xgboost>=1.7,<2.0" --quiet

# Verify
echo "Verifying installs..."
python3 -c "import pandas, numpy, torch; print('OK — CUDA:', torch.cuda.is_available(), '| Python:', __import__('sys').version)"
if [ $? -ne 0 ]; then
    echo "FATAL: package verification failed"
    aws s3 cp /var/log/user-data.log ${S3_BUCKET}/logs/training_pipeline_ecr.log
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
    exit 1
fi

# Background log sync every 60s
(
  while true; do
    [ -f "$OUT_DIR/training_pipeline_ecr.log" ] && \
      aws s3 cp $OUT_DIR/training_pipeline_ecr.log ${S3_BUCKET}/logs/training_pipeline_ecr.log > /dev/null 2>&1
    sleep 60
  done
) &
LOG_SYNC_PID=$!

# Run folds
echo "Starting Causal CFM-CVAE training..."
export PANEL_PATH=$WORK_DIR/biweekly_panel.csv
export OUT_DIR=$OUT_DIR

for FOLD in {0..8}; do
    echo "Starting Fold $FOLD..." | tee -a $OUT_DIR/training_pipeline_ecr.log
    python3 causal_cfm_cvae.py --fold $FOLD 2>&1 | tee -a $OUT_DIR/training_pipeline_ecr.log
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "FATAL: Fold $FOLD failed. Breaking." | tee -a $OUT_DIR/training_pipeline_ecr.log
        break
    fi
done

kill $LOG_SYNC_PID 2>/dev/null

# Final uploads
aws s3 cp $OUT_DIR/training_pipeline_ecr.log ${S3_BUCKET}/logs/training_pipeline_ecr.log
[ -f "$OUT_DIR/vae_dose_response_surface_expanded.csv" ] && \
  aws s3 cp $OUT_DIR/vae_dose_response_surface_expanded.csv ${S3_BUCKET}/output/vae_dose_response_surface_expanded.csv

# Self-terminate
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
