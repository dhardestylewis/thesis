import boto3

ec2 = boto3.client('ec2', region_name='us-east-1')
try:
    response = ec2.describe_instances()
    found = False
    for res in response.get('Reservations', []):
        for inst in res.get('Instances', []):
            name = next((t['Value'] for t in inst.get('Tags', []) if t['Key'] == 'Name'), 'Unknown')
            print(f"Instance: {name} | ID: {inst['InstanceId']} | State: {inst['State']['Name']} | IP: {inst.get('PublicIpAddress', 'None')}")
            found = True
    if not found:
        print("No EC2 instances found.")
except Exception as e:
    print('Error:', e)
