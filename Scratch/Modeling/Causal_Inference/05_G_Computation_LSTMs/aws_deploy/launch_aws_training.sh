#!/bin/bash
# ==============================================================================
# launch_aws_training.sh
# 
# Pushes the causal inference script and data to S3, and provides the exact 
# user-data script you can use to launch an EC2 instance that will automatically 
# run the container, sync results, and terminate itself.
# ==============================================================================

S3_BUCKET="s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline"
LOCAL_DIR="$(dirname $(pwd))" # Base directory: 05_G_Computation_LSTMs
DATA_FILE="${LOCAL_DIR}/../../../../../Data/Panel/biweekly_panel.csv"

echo "1. Syncing python scripts and Docker context to S3..."
aws s3 cp "${LOCAL_DIR}/causal_cfm_cvae.py" "${S3_BUCKET}/src/causal_cfm_cvae.py"
aws s3 cp Dockerfile "${S3_BUCKET}/src/Dockerfile"
aws s3 cp requirements.txt "${S3_BUCKET}/src/requirements.txt"

echo "2. Syncing dataset to S3..."
# Since biweekly_panel.csv is 97MB, it will upload quickly
aws s3 cp "$DATA_FILE" "${S3_BUCKET}/data/biweekly_panel.csv"

echo "======================================================================"
echo "SUCCESS: Payload staged on S3."
echo "======================================================================"
echo "To execute this unattended on a GPU instance (e.g. g4dn.xlarge), launch"
echo "an EC2 instance using the Deep Learning OSS Nvidia AMI and paste the"
echo "following block into the 'User Data' section:"
echo "======================================================================"
cat << 'EOF'
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
aws s3 cp ${S3_BUCKET}/data/ biweekly_panel.csv

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
EOF
