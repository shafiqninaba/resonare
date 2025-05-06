# Resonare – LLM Twin (Proof‑of‑Concept)

Fine‑tune an LLM on your personal chat history (Telegram, WhatsApp) with a simple Docker stack.

---

## 🛠 Prerequisites

| Tool / Service            | Version (minimum) | Notes |
|---------------------------|-------------------|-------|
| **Docker Desktop / Engine** | 24.x             | Linux, macOS or Windows WSL |
| **Docker Compose V2**       | bundled with Docker Desktop<br>(use `docker compose`, not `docker‑compose`) |
| **AWS account**           | – optional        | Only needed if you plan to push outputs to S3 (enabled by default). |
| **Git**                   | any recent       | To clone the repository. |

> **GPU support (optional)**  
> If you want to fine‑tune with CUDA: install the *NVIDIA Container Toolkit* and make sure `nvidia-smi` works on host.

---

## 🚀 Quick‑start

### 1  Clone the repo

```bash
git clone https://github.com/shafiqninaba/resonare.git
cd resonare
```

### 2  Create an `.env`

Copy `.env.example` (if present) or create a new `.env` at the project root:

```bash
# AWS – optional (required only for S3 uploads)
AWS_ACCESS_KEY_ID=XXXXXXXX
AWS_SECRET_ACCESS_KEY=YYYYYYYYYYYYYYYY
AWS_S3_BUCKET=resonare-test-bucket
AWS_REGION=ap-southeast-1

# Optional service‑to‑service overrides
DATA_PREP_URL=http://data-prep:8000
FINE_TUNING_URL=http://fine-tuning:8000
INFERENCE_URL=http://inference:8000
```

*Leave the AWS variables blank if you want to run completely locally.*

### 3  Build & run the stack

```bash
docker compose up --build
```

**What starts?**

| Service     | Tech          | Port  | Purpose                               |
| ----------- | ------------- | ----- | ------------------------------------- |
| `data-prep` | FastAPI       | :8000 | Parse chat JSON, chunk & upload to S3 |
| `fine-tune` | FastAPI + GPU | —     | Queue + run LoRA / Unsloth training   |
| `inference` | FastAPI       | :8001 | (future) Serve your trained model     |
| `frontend`  | Streamlit     | :8501 | Web UI – upload, monitor, launch jobs |

Open [http://localhost:8501](http://localhost:8501) to use Resonare.

### 4  Shut everything down

```bash
docker compose down
```

All containers and the default bridge network will be removed.
Your Docker images remain cached; next `up --build` will reuse layers.

---


## Project Layout

```
resonare/
├─ docker-compose.yml
├─ packages/
│  ├─ data-prep/      # FastAPI app + preprocessing logic
│  ├─ fine-tuning/    # Training worker (Unsloth / LoRA)
│  ├─ inference/      # (WIP) model serving
│  └─ frontend/       # Streamlit UI
└─ .env               # your environment variables
```

---
