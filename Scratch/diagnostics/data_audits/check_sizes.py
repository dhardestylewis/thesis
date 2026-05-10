import os

def get_stats(path):
    if not os.path.exists(path):
        return 0, 0
    files = [os.path.join(path, f) for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    total_size = sum(os.path.getsize(f) for f in files)
    return len(files), total_size / (1024 * 1024 * 1024)

comm_files, comm_size = get_stats(r'c:\Users\dhl\data\Thesis\thesis\Data\Commission_PDFs')
cc_files, cc_size = get_stats(r'c:\Users\dhl\data\Thesis\thesis\Data\Council_Minutes_PDFs')

print(f"--- DOWNLOAD STATISTICS ---")
print(f"Commissions PDFs Folder:")
print(f"  Files: {comm_files}")
print(f"  Size : {comm_size:.2f} GB\n")

print(f"City Council Minutes Folder:")
print(f"  Files: {cc_files}")
print(f"  Size : {cc_size:.2f} GB\n")

print(f"TOTAL HYDRATED DATASET:")
print(f"  Files: {comm_files + cc_files}")
print(f"  Size : {comm_size + cc_size:.2f} GB")
