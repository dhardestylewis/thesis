#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Updating system..."
yum update -y
yum install -y python3-pip git aws-cli tmux

# Paths
export WORK_DIR="/home/ec2-user/thesis_run"
export OUT_DIR="$WORK_DIR/artifacts"
mkdir -p $WORK_DIR $OUT_DIR
cd $WORK_DIR

# Pull assets from S3
echo "Downloading assets from S3..."
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/src/08e_run_multihorizon_oot.py .
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/data/biweekly_panel.csv .

# Create venv and install dependencies
echo "Installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy catboost scikit-learn tqdm pyarrow torch xgboost

# Verify all critical packages installed before running anything
echo "Verifying installs..."
python3 -c "import pandas, numpy, catboost, sklearn, torch, xgboost; print('All packages OK')"
if [ $? -ne 0 ]; then
    echo "FATAL: package verification failed — uploading install log and aborting"
    aws s3 cp /var/log/user-data.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
    INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
    exit 1
fi

# Background sync: push run.log + checkpoint CSV to S3 every 60s for live monitoring
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

# Kill sync loop
kill $LOG_SYNC_PID 2>/dev/null

# Final upload
echo "Uploading final results to S3..."
aws s3 cp $WORK_DIR/run.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log
[ -f "$OUT_DIR/multihorizon_multicutoff_all_models.csv" ] && \
  aws s3 cp $OUT_DIR/multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv

# IMDSv2-safe self-termination
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
