import subprocess
import json

res = subprocess.run(['aws.cmd', 'ec2', 'describe-instances', '--output', 'json'], capture_output=True)
data = json.loads(res.stdout)

for r in data['Reservations']:
    for i in r['Instances']:
        if i['State']['Name'] == 'running' and i['InstanceType'] == 'g4dn.xlarge':
            launch = i['LaunchTime']
            tags = {t['Key']: t['Value'] for t in i.get('Tags', [])}
            name = tags.get('Name', 'No Name')
            profile = i.get('IamInstanceProfile', {}).get('Arn', 'None')
            print(f"Instance: {i['InstanceId']} | Name: {name} | Launched: {launch} | Profile: {profile}")
