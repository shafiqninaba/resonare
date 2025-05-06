# Resonare – LLM Twin
Fine‑tune an LLM on your personal chat history (Telegram, WhatsApp).

## Prerequisites
TBC

## Quickstart
1. Clone the repository:
```bash
git clone https://github.com/shafiqninaba/resonare.git
cd resonare
```

2. Create an .env file in the root directory and add the following environment variables:
```
AWS_ACCESS_KEY_ID=XXXXXXXX
AWS_SECRET_ACCESS_KEY=YYYYYYYYYYYYYYYY
AWS_S3_BUCKET=resonare-test-bucket
AWS_REGION=ap-southeast-1
```

3. Build and run the docker compose file:
```bash
docker-compose up --build
```

4. Open your browser and go to `http://localhost:8501` to access the web app.


5. Close the web app and stop the docker container:
```bash
docker-compose down
```

## Project Structure
```bash
├── docker-compose.yml
├── main.py
├── packages
│   ├── data-prep
│   ├── fine-tuning
│   ├── frontend
│   └── inference
├── pyproject.toml
├── README.md
└── uv.lock
```

## Architecture

### Services
| Component     | Port   | Description                                        |
| ------------- | ------ | -------------------------------------------------- |
| **frontend**  | `8501` | Web UI – upload chats, configure jobs, monitor     |
| **data-prep** | `8000` | Parses Telegram JSON, chunks into LLM-ready blocks |
| **fine-tune** | —      | Queues and executes training with Unsloth          |
| **inference** | `8001` | Serves the fine-tuned model                        |


