#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Updating system..."
yum update -y
yum install -y python3-pip git aws-cli tmux

# Create working directory
export WORK_DIR="/home/ec2-user/thesis_run"
export OUT_DIR="$WORK_DIR/output"
mkdir -p $WORK_DIR
mkdir -p $OUT_DIR
cd $WORK_DIR

# Sync from S3
echo "Downloading assets from S3..."
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/src/08e_run_multihorizon_oot.py .
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/data/biweekly_panel.csv .

# Install requirements
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pandas numpy catboost scikit-learn tqdm pyarrow torch xgboost

# Run Pipeline
echo "Starting Multi-Horizon Pipeline on GPUs..."
export AWS_EXECUTION=1
python 08e_run_multihorizon_oot.py > $OUT_DIR/run.log 2>&1

# Upload output back to S3
echo "Uploading results to S3..."
aws s3 cp $OUT_DIR/run.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/multihorizon_run.log
if [ -f "multihorizon_multicutoff_all_models.csv" ]; then
    aws s3 cp multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv
fi
if [ -f "artifacts/multihorizon_multicutoff_all_models.csv" ]; then
    aws s3 cp artifacts/multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv
fi
if [ -f "$OUT_DIR/multihorizon_multicutoff_all_models.csv" ]; then
    aws s3 cp $OUT_DIR/multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/multihorizon_multicutoff_all_models.csv
fi

# Terminate instance to save costs
TOKEN=`curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null`
INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
