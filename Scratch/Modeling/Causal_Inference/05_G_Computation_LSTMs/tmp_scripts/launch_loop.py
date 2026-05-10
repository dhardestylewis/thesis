import time
import subprocess
import json

while True:
    # Run AWS CLI command to describe instances
    res = subprocess.run(['aws.cmd', 'ec2', 'describe-instances', '--query', 'Reservations[*].Instances[*].[InstanceId,State.Name,InstanceType]', '--output', 'json'], capture_output=True)
    
    if res.returncode == 0:
        try:
            data = json.loads(res.stdout)
            running_g4 = sum([1 for r in data for i in r if i[1] == 'running' and i[2] == 'g4dn.xlarge'])
            shutting_g4 = sum([1 for r in data for i in r if i[1] == 'shutting-down' and i[2] == 'g4dn.xlarge'])
            
            if running_g4 < 2 and shutting_g4 == 0:
                print('vCPU limit cleared! Launching instance...', flush=True)
                cmd = [
                    'aws.cmd', 'ec2', 'run-instances', 
                    '--image-id', 'ami-0ad40b42b2dc8ba2a', 
                    '--instance-type', 'g4dn.xlarge', 
                    '--security-group-ids', 'sg-0f871eaf0e651f50d', 
                    '--iam-instance-profile', 'Arn=arn:aws:iam::903537643799:instance-profile/homecastr-backtest-ec2', 
                    '--key-name', 'thesis-causal-debug', 
                    '--user-data', 'file://aws_deploy/user-data.sh'
                ]
                out = subprocess.run(cmd, capture_output=True)
                print(out.stdout.decode('utf-8'))
                if out.stderr:
                    print('Error:', out.stderr.decode('utf-8'))
                break
            else:
                print(f'Waiting... {running_g4} running, {shutting_g4} shutting-down.', flush=True)
                time.sleep(15)
        except Exception as e:
            print('JSON Parsing Error:', e)
            time.sleep(15)
    else:
        print('AWS CLI Error:', res.stderr.decode('utf-8'))
        time.sleep(15)
