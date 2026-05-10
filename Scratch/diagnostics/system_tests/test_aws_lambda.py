import boto3
import json
import zipfile
import io
import time

lambda_client = boto3.client('lambda', region_name='us-east-1')
iam_client = boto3.client('iam')

function_name = 'AustinEdimsBenchmark'
role_name = 'lambda-basic-execution-role-edims'

print("Setting up IAM role for Lambda...")
try:
    role_response = iam_client.get_role(RoleName=role_name)
    role_arn = role_response['Role']['Arn']
except iam_client.exceptions.NoSuchEntityException:
    assume_role_policy = {
        "Version": "2012-10-17",
        "Statement": [{"Action": "sts:AssumeRole", "Principal": {"Service": "lambda.amazonaws.com"}, "Effect": "Allow"}]
    }
    role_response = iam_client.create_role(RoleName=role_name, AssumeRolePolicyDocument=json.dumps(assume_role_policy))
    role_arn = role_response['Role']['Arn']
    iam_client.attach_role_policy(RoleName=role_name, PolicyArn='arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole')
    print("Created new role, waiting 10s for propagation...")
    time.sleep(10)

code = """
import urllib.request
import concurrent.futures
import time

urls = [
    'https://services.austintexas.gov/edims/document.cfm?id=472409',
    'https://services.austintexas.gov/edims/document.cfm?id=472547',
    'https://services.austintexas.gov/edims/document.cfm?id=472548',
    'https://services.austintexas.gov/edims/document.cfm?id=471473',
    'https://services.austintexas.gov/edims/document.cfm?id=471989',
    'https://services.austintexas.gov/edims/document.cfm?id=472030',
    'https://services.austintexas.gov/edims/document.cfm?id=471969',
    'https://services.austintexas.gov/edims/document.cfm?id=471968',
    'https://services.austintexas.gov/edims/document.cfm?id=471476',
    'https://services.austintexas.gov/edims/document.cfm?id=471967',
] * 2

def download_pdf(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as response:
            data = response.read()
            return True, len(data)
    except Exception as e:
        return False, str(e)

def test_threads(thread_count, url_list):
    start_time = time.time()
    success = 0
    errors = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = list(executor.map(download_pdf, url_list))
    for res, info in results:
        if res: success += 1
        else: errors[info] = errors.get(info, 0) + 1
    elapsed = time.time() - start_time
    return f"Threads: {thread_count} | Success: {success}/20 | Time: {elapsed:.2f}s | Errors: {errors}"

def lambda_handler(event, context):
    r1 = test_threads(5, urls[:20])
    time.sleep(3)
    r2 = test_threads(10, urls[:20])
    time.sleep(3)
    r3 = test_threads(20, urls[:20])
    return [r1, r2, r3]
"""

print("Creating Lambda Deployment Package...")
zip_buffer = io.BytesIO()
with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
    zip_file.writestr('lambda_function.py', code)
zip_buffer.seek(0)

print("Deploying Lambda Function...")
try:
    lambda_client.delete_function(FunctionName=function_name)
    time.sleep(2)
except lambda_client.exceptions.ResourceNotFoundException:
    pass

retry_count = 0
while retry_count < 5:
    try:
        lambda_client.create_function(
            FunctionName=function_name,
            Runtime='python3.9',
            Role=role_arn,
            Handler='lambda_function.lambda_handler',
            Code={'ZipFile': zip_buffer.read()},
            Timeout=120,
            MemorySize=512
        )
        break
    except lambda_client.exceptions.InvalidParameterValueException as e:
        print("Role not fully propagated, retrying in 5s...")
        time.sleep(5)
        retry_count += 1
        zip_buffer.seek(0)

print("Invoking Lambda Function on AWS...")
# Wait for active state
time.sleep(5)
response = lambda_client.invoke(
    FunctionName=function_name,
    InvocationType='RequestResponse'
)

payload = json.loads(response['Payload'].read().decode('utf-8'))
print("\n--- AWS BENCHMARK RESULTS ---")
if isinstance(payload, list):
    for r in payload:
        print(r)
else:
    print(payload)

print("\nCleaning up Lambda...")
lambda_client.delete_function(FunctionName=function_name)
print("Done!")
