#!/bin/bash
exec > >(tee /var/log/user-data.log | logger -t user-data -s 2>/dev/console) 2>&1

echo "=== PYTHON PATH PROBE ==="
echo "which python:"; which python 2>&1
echo "which python3:"; which python3 2>&1
echo "find conda pythons:"; find / -name "python" -o -name "python3" 2>/dev/null | grep -v proc | grep -v sys | head -20
echo "find conda:"; find /usr /opt /home -name "conda" 2>/dev/null | head -10
echo "PATH=$PATH"
echo "=== PROBE DONE ==="

# Upload result to S3
aws s3 cp /var/log/user-data.log s3://mineflow-v3-horizon-1ed4ab27/thesis-pipeline/logs/probe.log

INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
