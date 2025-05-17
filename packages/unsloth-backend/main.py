import hydra
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from omegaconf import OmegaConf

from app.dependencies import lifespan, logger
from app.routers import health, jobs, inference

# Load environment variables
load_dotenv()

# Create the FastAPI app
app = FastAPI(
    title="AI Model API",
    description="API for model fine-tuning and inference",
    lifespan=lifespan,
)

# Include routers
app.include_router(jobs.router)
app.include_router(inference.router)
app.include_router(health.router)

if __name__ == "__main__":
    try:
        with hydra.initialize(config_path="conf"):
            cfg = hydra.compose(config_name="config")

        logging_config = OmegaConf.to_container(cfg.logging, resolve=True)
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000, log_config=logging_config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        uvicorn.run("app.main:app", host="0.0.0.0", port=8000)