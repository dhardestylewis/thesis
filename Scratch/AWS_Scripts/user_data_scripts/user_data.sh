#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "=== Starting Multi-Horizon Pipeline Setup ==="

# Paths
export WORK_DIR="/home/ec2-user/thesis_run"
export OUT_DIR="$WORK_DIR/artifacts"
mkdir -p $WORK_DIR $OUT_DIR
cd $WORK_DIR

# Ensure python3 and pip are installed (for Ubuntu 22.04)
echo "Installing system dependencies..."
sudo apt-get update -y
sudo apt-get install -y python3 python3-pip python3-venv awscli

# Pull assets from S3
echo "Downloading assets from S3..."
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/src/08e_run_multihorizon_oot.py .
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/data/biweekly_panel.csv .

# Install packages directly into system Python (avoids venv setuptools conflicts)
echo "Installing required packages via pip..."
pip3 install --upgrade pip --quiet
pip3 install --upgrade "setuptools<70" wheel --quiet
pip3 install pandas numpy scikit-learn pyarrow --quiet
pip3 install torch --index-url https://download.pytorch.org/whl/cpu --quiet
pip3 install "catboost>=1.2,<1.3" --quiet  # 1.2.x ships pre-built wheels, no setuptools>=70 dep
pip3 install "xgboost>=1.7,<2.0" --quiet  # 1.7.x ships pre-built wheels, no cmake needed

# Verify
echo "Verifying installs..."
python3 -c "import pandas, numpy, catboost, sklearn, torch, xgboost; print('All packages OK — Python:', __import__('sys').version)"
if [ $? -ne 0 ]; then
    echo "FATAL: package verification failed"
    aws s3 cp /var/log/user-data.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
    exit 1
fi

# Use python3 for the run script too
alias python=python3

# Background log sync every 60s
(
  while true; do
    [ -f "$WORK_DIR/run.log" ] && \
      aws s3 cp $WORK_DIR/run.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log > /dev/null 2>&1
    [ -f "$OUT_DIR/multihorizon_multicutoff_all_models.csv" ] && \
      aws s3 cp $OUT_DIR/multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv > /dev/null 2>&1
    sleep 60
  done
) &
LOG_SYNC_PID=$!

# Run pipeline
echo "Starting Multi-Horizon Pipeline..."
export AWS_EXECUTION=1
python3 08e_run_multihorizon_oot.py > $WORK_DIR/run.log 2>&1

kill $LOG_SYNC_PID 2>/dev/null

# Final upload
echo "Uploading results..."
aws s3 cp $WORK_DIR/run.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log
[ -f "$OUT_DIR/multihorizon_multicutoff_all_models.csv" ] && \
  aws s3 cp $OUT_DIR/multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv

# IMDSv2-safe self-termination
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
