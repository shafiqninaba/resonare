import torch


def main():
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")


if __name__ == "__main__":
    main()
