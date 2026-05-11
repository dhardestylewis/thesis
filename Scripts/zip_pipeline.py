import os
import zipfile

def zip_pipeline():
    with zipfile.ZipFile('pipeline_code.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ['.git', '.venv', '.venv_cuda', '.clone', 'Data', 'Archive', 'logs', 'output', 'catboost_info', '__pycache__']]
            for file in files:
                if file.endswith('.zip') or file.endswith('.csv') or file.endswith('.pdf') or file.endswith('.mp4'):
                    continue
                file_path = os.path.join(root, file)
                zipf.write(file_path, file_path)

if __name__ == '__main__':
    zip_pipeline()
    print("Done zipping pipeline code!")
