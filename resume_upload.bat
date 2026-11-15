@echo off
cd /d "C:\Users\stream\Desktop\m2 pfe\MASTER2-A1"
for /f "usebackq delims=" %%k in (`python -c "import json;print(json.load(open(r'C:\Users\stream\.kaggle\kaggle.json'))['key'])"`) do set KAGGLE_API_TOKEN=%%k
python scripts\kaggle_upload_single_version.py