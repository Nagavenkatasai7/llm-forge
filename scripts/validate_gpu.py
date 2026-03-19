"""Validate GPU setup and print diagnostics."""

import sys


def main():
    print("=== GPU Validation ===\n")

    # Check CUDA
    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                vram_gb = props.total_mem / (1024**3)
                print(f"\nGPU {i}: {props.name}")
                print(f"  VRAM: {vram_gb:.1f} GB")
                print(f"  Compute capability: {props.major}.{props.minor}")
                print(f"  Multi-processor count: {props.multi_processor_count}")

            # Test allocation
            print("\nTesting GPU memory allocation...")
            x = torch.randn(1000, 1000, device="cuda:0")
            print(f"  Allocated 1000x1000 tensor on GPU: OK")
            del x
            torch.cuda.empty_cache()
            print("  Memory cleared: OK")
        else:
            print("\nNo CUDA GPU available. Training will use CPU (slow).")
            print("For GPU training, install CUDA toolkit and PyTorch with CUDA support.")

    except ImportError:
        print("ERROR: PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    # Check transformers
    try:
        import transformers

        print(f"\ntransformers version: {transformers.__version__}")
    except ImportError:
        print("\nWARNING: transformers not installed")

    # Check flash attention
    try:
        import flash_attn

        print(f"flash-attn version: {flash_attn.__version__}")
    except ImportError:
        print("flash-attn: not installed (optional, recommended for speedup)")

    print("\n=== Validation Complete ===")


if __name__ == "__main__":
    main()
