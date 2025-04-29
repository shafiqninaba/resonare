from src.general_utils import setup_logger, setup_s3_client

if __name__ == "__main__":
    # Set up logger
    logger = setup_logger("main")

    logger.debug("This is a debug message.")
    logger.info("This is an info message.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
    logger.critical("This is a critical message.")

    setup_s3_client(None)  # Pass None for cfg as a placeholder
