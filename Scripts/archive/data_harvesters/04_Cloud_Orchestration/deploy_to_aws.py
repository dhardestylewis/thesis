import os
import boto3
import time
import sys

def deploy_aws_pipeline(access_key, secret_key, region='us-east-1'):
    print("[*] Initializing AWS Boto3 Client...")
    ec2 = boto3.client('ec2', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)
    ec2_resource = boto3.resource('ec2', aws_access_key_id=access_key, aws_secret_access_key=secret_key, region_name=region)

    key_name = "homecastr-pipeline-key"
    
    # 1. Create KeyPair if it doesn't exist
    print(f"[*] Checking for existing SSH KeyPair '{key_name}'...")
    try:
        key_pair = ec2.create_key_pair(KeyName=key_name)
        with open(f"{key_name}.pem", "w") as file:
            file.write(key_pair['KeyMaterial'])
        os.chmod(f"{key_name}.pem", 0o400)
        print(f"[+] Created new KeyPair and saved to {key_name}.pem")
    except ec2.exceptions.ClientError as e:
        if 'InvalidKeyPair.Duplicate' in str(e):
            print(f"[*] KeyPair '{key_name}' already exists in AWS.")
        else:
            raise e

    # 2. Get default VPC and create a permissive Security Group
    vpcs = ec2.describe_vpcs()
    vpc_id = vpcs['Vpcs'][0]['VpcId']
    sg_name = "homecastr-pipeline-sg"
    
    try:
        sg_response = ec2.create_security_group(GroupName=sg_name, Description='Allow SSH', VpcId=vpc_id)
        sg_id = sg_response['GroupId']
        ec2.authorize_security_group_ingress(
            GroupId=sg_id,
            IpPermissions=[{'IpProtocol': 'tcp', 'FromPort': 22, 'ToPort': 22, 'IpRanges': [{'CidrIp': '0.0.0.0/0'}]}]
        )
        print(f"[+] Created Security Group {sg_name} with SSH access.")
    except ec2.exceptions.ClientError as e:
        if 'InvalidGroup.Duplicate' in str(e):
            sgs = ec2.describe_security_groups(GroupNames=[sg_name])
            sg_id = sgs['SecurityGroups'][0]['GroupId']
            print(f"[*] Security Group '{sg_name}' already exists.")
        else:
            raise e

    # 3. Spin up the EC2 Instance (Ubuntu 22.04 LTS on t3.small)
    print("[*] Provisioning EC2 Instance (t3.small)...")
    ami_id = "ami-0c7217cdde317cfec" # Ubuntu 22.04 LTS in us-east-1
    instances = ec2_resource.create_instances(
        ImageId=ami_id,
        MinCount=1,
        MaxCount=1,
        InstanceType="t3.small",
        KeyName=key_name,
        SecurityGroupIds=[sg_id]
    )
    
    instance = instances[0]
    print(f"[*] Instance {instance.id} created. Waiting for it to enter 'running' state...")
    instance.wait_until_running()
    instance.reload()
    
    public_ip = instance.public_ip_address
    print(f"\n[+] SUCCESS! AWS EC2 Node is live at IP: {public_ip}")
    print(f"\n[*] Please wait ~60 seconds for the SSH daemon to initialize before connecting.")
    print(f"ssh -i {key_name}.pem ubuntu@{public_ip}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python deploy_to_aws.py <ACCESS_KEY> <SECRET_KEY>")
        sys.exit(1)
        
    deploy_aws_pipeline(sys.argv[1], sys.argv[2])
