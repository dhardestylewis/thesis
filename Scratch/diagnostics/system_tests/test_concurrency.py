import pandas as pd
import urllib.request
import concurrent.futures
import time

index_csv = r"c:\Users\dhl\data\Thesis\thesis\Data\planning_commission_index.csv"
df = pd.read_csv(index_csv)
df_staff = df[df['Doc_Text'].str.contains('Staff Report', case=False, na=False)]
# Sample 100 random urls for our concurrency test
urls = df_staff['Doc_URL'].sample(100, random_state=42).tolist()

def download_pdf(url):
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
            return True, len(data)
    except Exception as e:
        return False, str(e)

def test_threads(thread_count, url_list):
    print(f"\n--- Testing with {thread_count} concurrent threads ---")
    start_time = time.time()
    success = 0
    errors = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=thread_count) as executor:
        results = list(executor.map(download_pdf, url_list))
        
    for res, info in results:
        if res:
            success += 1
        else:
            errors[info] = errors.get(info, 0) + 1
            
    elapsed = time.time() - start_time
    print(f"Completed 20 requests in {elapsed:.2f} seconds.")
    print(f"Success rate: {success}/20 ({(success/20)*100:.1f}%)")
    if errors:
        print("Errors encountered:")
        for err, count in errors.items():
            print(f"  {err}: {count} times")

print("Starting Local Concurrency Benchmark on Austin EDIMS...")
test_threads(5, urls[0:20])
time.sleep(5) # Give the server a breather
test_threads(10, urls[20:40])
time.sleep(5)
test_threads(20, urls[40:60])
time.sleep(5)
test_threads(40, urls[60:80])
