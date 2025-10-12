@echo off
setlocal
call .venv\Scripts\activate
pip install -r requirements.txt
python -m bin.generate_data --out out\data --frames 40 --noise 0.7
python -m bin.solve_extrinsics --cams out\data\cams.json --detections out\data\detections.csv --out out\results.json
echo Done. See out\results.json
endlocal
