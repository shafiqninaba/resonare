fastapi dev app:app --host 0.0.0.0 --port 8010
uvicorn app:app --host 0.0.0.0 --port 8010 --reload

# upload the merged JSON produced by some script
curl -X POST -H "Content-Type: application/json" --data @/Users/lowrenhwa/Desktop/resonare/packages/data-prep/data/raw/result.json \
     http://localhost:8010/data-prep/process

# check a single job
curl http://localhost:8010/data-prep/49613daf0b144f239b1983561ec180a0/status

# see full queue
curl http://localhost:8010/data-prep/queue

# check if s3 client is up
curl http://localhost:8010/data-prep/health