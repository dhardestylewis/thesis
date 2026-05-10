#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Updating system..."
yum update -y
yum install -y python3-pip git aws-cli tmux

export WORK_DIR="/home/ec2-user/thesis_run"
export OUT_DIR="$WORK_DIR/output"
mkdir -p $WORK_DIR
mkdir -p $OUT_DIR
cd $WORK_DIR

echo "Downloading assets from S3..."
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/src/08e_run_annualized_oot.py .
aws s3 cp s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/data/biweekly_panel.csv .

pip3 install pandas numpy catboost scikit-learn tqdm pyarrow torch xgboost

echo "Starting Annualized Pipeline..."
python3 08e_run_annualized_oot.py > $OUT_DIR/run.log 2>&1

echo "Uploading results to S3..."
aws s3 cp $OUT_DIR/run.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/annualized_run.log

if [ -f "artifacts/annualized_multihorizon_multicutoff_all_models.csv" ]; then
    aws s3 cp artifacts/annualized_multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/annualized_multihorizon_multicutoff_all_models.csv
fi
if [ -f "annualized_multihorizon_multicutoff_all_models.csv" ]; then
    aws s3 cp annualized_multihorizon_multicutoff_all_models.csv s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/output/annualized_multihorizon_multicutoff_all_models.csv
fi

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region us-east-1
