"""Download a base model from HuggingFace Hub."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="Download a base model from HuggingFace Hub")
    parser.add_argument("model_name", help="HuggingFace model ID (e.g., meta-llama/Llama-3.2-1B)")
    parser.add_argument("--cache-dir", default=None, help="Cache directory for model files")
    parser.add_argument("--revision", default=None, help="Model revision/branch")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading model: {args.model_name}")
        path = snapshot_download(
            args.model_name,
            revision=args.revision,
            cache_dir=args.cache_dir,
        )
        print(f"Model downloaded to: {path}")

    except ImportError:
        print("ERROR: huggingface_hub is required. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed to download model: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
