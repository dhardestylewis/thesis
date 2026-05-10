#!/bin/bash
# Install Docker if not present
sudo yum update -y
sudo amazon-linux-extras install docker -y
sudo service docker start
sudo usermod -a -G docker ec2-user

# Define Paths
S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
WORK_DIR="/home/ec2-user/causal_run"
OUT_DIR="$WORK_DIR/output"

mkdir -p $WORK_DIR
mkdir -p $OUT_DIR
cd $WORK_DIR

# Pull assets
aws s3 cp ${S3_BUCKET}/src/ . --recursive
aws s3 cp ${S3_BUCKET}/data/biweekly_panel.csv biweekly_panel.csv

# Build Container
sudo docker build -t causal_pipeline .

# Start a background process to continuously sync logs to S3 every 60 seconds
aws s3 cp /dev/null ${S3_BUCKET}/logs/training_pipeline.log
(
  while true; do
    if [ -f "$OUT_DIR/training_pipeline.log" ]; then
      aws s3 cp $OUT_DIR/training_pipeline.log ${S3_BUCKET}/logs/training_pipeline.log > /dev/null 2>&1
    fi
    sleep 60
  done
) &
LOG_SYNC_PID=$!

# Run Container, piping output to a log file
sudo docker run --gpus all \
    -v $(pwd):/data \
    -e PANEL_PATH=/data/biweekly_panel.csv \
    -e OUT_DIR=/data/output \
    causal_pipeline 2>&1 | tee $OUT_DIR/training_pipeline.log

# Kill the background log sync loop once training is complete
kill $LOG_SYNC_PID

# Upload God Table back to S3 (if it exists)
if [ -f "$OUT_DIR/vae_dose_response_surface_expanded.csv" ]; then
    aws s3 cp $OUT_DIR/vae_dose_response_surface_expanded.csv ${S3_BUCKET}/output/vae_dose_response_surface_expanded.csv
fi

# Always upload the logs to S3
aws s3 cp $OUT_DIR/training_pipeline.log ${S3_BUCKET}/logs/training_pipeline.log

# Upload Model Weights to S3
aws s3 cp $OUT_DIR/ ${S3_BUCKET}/output/ --recursive --exclude "*" --include "causal_cfm_weights_fold_*.pt"

# Terminate instance to save costs
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
INSTANCE_ID=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
