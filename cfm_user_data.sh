#!/bin/bash
# Install Docker if not present
if ! command -v docker &> /dev/null; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io
    sudo systemctl start docker
    sudo usermod -a -G docker ubuntu
fi

# Define Paths
S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
WORK_DIR="/home/ubuntu/causal_run"
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
for FOLD in {0..8}; do
    echo "Starting Fold $FOLD..." | tee -a $OUT_DIR/training_pipeline.log
    sudo docker run --gpus all \
        -v $(pwd):/data \
        -e PANEL_PATH=/data/biweekly_panel.csv \
        -e OUT_DIR=/data/output \
        causal_pipeline python causal_cfm_cvae.py --fold $FOLD 2>&1 | tee -a $OUT_DIR/training_pipeline.log
done

# Kill the background log sync loop once training is complete
kill $LOG_SYNC_PID

# Upload God Table back to S3 (if it exists)
if [ -f "$OUT_DIR/vae_dose_response_surface_expanded.csv" ]; then
    aws s3 cp $OUT_DIR/vae_dose_response_surface_expanded.csv ${S3_BUCKET}/output/vae_dose_response_surface_expanded.csv
fi

# Always upload the logs to S3
aws s3 cp $OUT_DIR/training_pipeline.log ${S3_BUCKET}/logs/training_pipeline.log

# Terminate instance to save costs
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
