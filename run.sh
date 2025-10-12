(
echo #!/usr/bin/env bash
echo set -e
echo python -m venv .venv
echo source .venv/bin/activate
echo pip install -r requirements.txt
echo python -m bin.generate_data --out out/data --frames 40 --noise 0.7
echo python -m bin.solve_extrinsics --cams out/data/cams.json --detections out/data/detections.csv --out out/results.json
echo echo "Done. See out/results.json"
) > run.sh
