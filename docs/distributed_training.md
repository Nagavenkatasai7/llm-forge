# Distributed Training Guide

Scale training across multiple GPUs and nodes with llm-forge's built-in distributed training framework. This guide covers when to use distributed training, how each framework works, configuration patterns, and troubleshooting.

---

## Table of Contents

1. [When You Need Distributed Training](#when-you-need-distributed-training)
2. [Memory Math: Why Models Do Not Fit](#memory-math-why-models-do-not-fit)
3. [Hardware Profiler](#hardware-profiler)
4. [FSDP (Fully Sharded Data Parallel)](#fsdp-fully-sharded-data-parallel)
5. [DeepSpeed ZeRO (Stages 0-3)](#deepspeed-zero-stages-0-3)
6. [Megatron-Core: 3D Parallelism](#megatron-core-3d-parallelism)
7. [FP8 Training on Hopper GPUs](#fp8-training-on-hopper-gpus)
8. [Multi-GPU Setup on a Single Machine](#multi-gpu-setup-on-a-single-machine)
9. [Multi-Node Training with SLURM](#multi-node-training-with-slurm)
10. [Framework Selection Decision Guide](#framework-selection-decision-guide)
11. [Example Configs for Each Framework](#example-configs-for-each-framework)
12. [Troubleshooting](#troubleshooting)

---

## When You Need Distributed Training

A single GPU can only hold so much data. During training, GPU VRAM must store:

- **Model weights** -- the parameters of the neural network.
- **Gradients** -- same size as the weights, computed during the backward pass.
- **Optimizer states** -- AdamW stores a momentum vector and a variance vector in fp32, plus an fp32 copy of the weights. That is 12 bytes per parameter.
- **Activations** -- intermediate values from the forward pass, needed for the backward pass. These grow with batch size and sequence length.

For a 7B-parameter model with mixed-precision AdamW, the minimum memory budget (before activations) is roughly:

```
Weights:          7B * 2 bytes    = 14 GB   (bf16)
Gradients:        7B * 2 bytes    = 14 GB   (bf16)
Optimizer states: 7B * 12 bytes   = 84 GB   (fp32 master + momentum + variance)
----------------------------------------------
Subtotal (no activations):         112 GB
```

That exceeds the 80 GB capacity of an A100. You need distributed training to shard these components across multiple GPUs.

**Rules of thumb:**

| Model Size | Single GPU Feasible?                                | Recommended Approach                    |
|------------|-----------------------------------------------------|-----------------------------------------|
| < 1B       | Yes, full fine-tune on a 24 GB GPU                  | Single GPU, no distribution needed      |
| 1B - 3B    | Yes with LoRA; tight for full fine-tune              | Single GPU with LoRA/QLoRA              |
| 3B - 8B    | LoRA only on 24 GB; full fine-tune needs 80 GB+     | FSDP or DeepSpeed ZeRO-2 on multi-GPU  |
| 8B - 20B   | Needs multiple GPUs even for LoRA full forward pass  | DeepSpeed ZeRO-3 or FSDP FULL_SHARD    |
| 20B - 70B  | Needs model parallelism + data parallelism           | DeepSpeed ZeRO-3 with tensor parallelism|
| > 70B      | Needs full 3D parallelism across many nodes          | Megatron-Core TP + PP + DP             |

---

## Memory Math: Why Models Do Not Fit

llm-forge includes a `HardwareProfiler` class that performs these calculations precisely. Understanding the math helps you reason about configuration choices.

### Bytes Per Parameter by Precision

| Precision | Bytes / Param | Description                                 |
|-----------|---------------|---------------------------------------------|
| fp32      | 4.0           | Full 32-bit floating point                  |
| fp16      | 2.0           | Half precision                              |
| bf16      | 2.0           | Brain floating point (Ampere+)              |
| fp8       | 1.0           | 8-bit floating point (Hopper+)              |
| int8      | 1.0           | 8-bit integer quantization                  |
| int4/nf4  | 0.5           | 4-bit quantization (QLoRA)                  |

### Optimizer State Sizes

| Optimizer            | Bytes / Trainable Param | Notes                           |
|----------------------|------------------------|---------------------------------|
| `adamw_torch`        | 12.0                   | fp32 copy + momentum + variance |
| `adamw_8bit`         | 6.0                    | 8-bit optimizer states          |
| `paged_adamw_32bit`  | 12.0                   | Paged to CPU on OOM             |
| `paged_adamw_8bit`   | 6.0                    | Paged + 8-bit states            |
| `sgd`                | 4.0                    | Momentum only                   |
| `adafactor`          | 8.0                    | Row + column factored states    |

### LoRA Memory Savings

With LoRA, only adapter parameters are trainable (typically 1-3% of the total). The base model weights are frozen and stored in the inference precision (e.g. bf16 or int4). Gradients and optimizer states apply only to the trainable LoRA parameters. For a 7B model with LoRA rank 16:

```
Trainable params:   ~2% of 7B = 140M
Gradients:          140M * 2 bytes = 0.28 GB
Optimizer states:   140M * 12 bytes = 1.68 GB
Base model weights: 7B * 2 bytes = 14 GB (frozen, bf16)
LoRA weights:       140M * 2 bytes = 0.28 GB
-----------------------------------------------
Total (no activations):  ~16.2 GB   (fits on a 24 GB GPU)
```

---

## Hardware Profiler

llm-forge provides a `HardwareProfiler` in `llm_forge.training.distributed.hardware_profiler` that estimates memory requirements and recommends a parallelism strategy.

### Programmatic Usage

```python
from llm_forge.training.distributed.hardware_profiler import HardwareProfiler

profiler = HardwareProfiler()

# Estimate memory for a 7B model, full fine-tune, bf16, on 4 GPUs with ZeRO-2
estimate = profiler.estimate_memory(
    model_params=7.0,           # Billions
    precision="bf16",
    optimizer="adamw_torch",
    num_gpus=4,
    zero_stage=2,
    tensor_parallel=1,
    pipeline_parallel=1,
    lora_rank=None,             # Full fine-tune (no LoRA)
)
print(estimate)
# MemoryEstimate(total=112.00 GB, per_gpu=35.00 GB, gpus=4, strategy=ZeRO-2)

# Same model with LoRA rank 16
estimate_lora = profiler.estimate_memory(
    model_params=7.0,
    precision="bf16",
    optimizer="adamw_torch",
    num_gpus=1,
    zero_stage=0,
    lora_rank=16,
    lora_fraction=0.02,
)
print(estimate_lora)
# MemoryEstimate(total=16.24 GB, per_gpu=16.24 GB, gpus=1, strategy=none)
```

### Recommendation Engine

The `DistributedOrchestrator` wraps the profiler and selects the framework automatically:

```python
from llm_forge.training.distributed.orchestrator import DistributedOrchestrator

orchestrator = DistributedOrchestrator(config)
framework = orchestrator.select_framework(
    model_size_b=13.0,
    num_gpus=8,
    gpu_type="A100",
    vram_per_gpu=80.0,
)
print(framework)   # "deepspeed"

# Generate the framework-specific configuration
framework_config = orchestrator.generate_config(
    framework=framework,
    model_size_b=13.0,
    num_gpus=8,
    vram_per_gpu=80.0,
)
```

### Automatic Selection Matrix

The orchestrator uses this decision matrix internally:

| Model Size | GPU Count | Recommended Framework         |
|------------|-----------|-------------------------------|
| <= 1B      | any       | FSDP `NO_SHARD` (DDP)        |
| 1B - 8B    | 1-8       | FSDP `FULL_SHARD`            |
| 8B - 20B   | any       | DeepSpeed ZeRO-3             |
| 20B - 70B  | 2+        | DeepSpeed ZeRO-3 + TP (+ PP) |
| > 70B      | 8+        | Megatron-Core 3D parallelism |

When you set `framework: "auto"` in your config, llm-forge's `auto_optimize_config()` and the orchestrator collaborate to pick the right strategy. On machines with NVLink, FSDP is preferred (lower communication overhead). Without NVLink, DeepSpeed is preferred (more tolerant of PCIe bandwidth).

---

## FSDP (Fully Sharded Data Parallel)

FSDP is PyTorch's native sharding solution, integrated via HuggingFace's `accelerate` library. It shards model parameters, gradients, and optimizer states across GPUs, then gathers them on demand for forward and backward computation.

### When to Use FSDP

- You have 2-8 GPUs on a single node with NVLink interconnects.
- Your model is 1B - 13B parameters.
- You want minimal configuration complexity (native PyTorch, no external runtime).
- You are using LoRA and need sharded base model weights.

### Sharding Strategies

| Strategy          | What Gets Sharded                          | Memory Savings | Communication Cost | When to Use                                         |
|-------------------|--------------------------------------------|----------------|--------------------|-----------------------------------------------------|
| `NO_SHARD`        | Nothing (pure DDP)                         | None           | Lowest             | Model fits in one GPU; you want data parallelism    |
| `SHARD_GRAD_OP`   | Gradients + optimizer states               | Moderate       | Low                | Model nearly fits; need ~2x optimizer memory savings|
| `FULL_SHARD`      | Parameters + gradients + optimizer states  | Maximum        | Moderate           | Model does not fit; need maximum per-GPU savings    |
| `HYBRID_SHARD`    | Full shard within node, replicate across   | High           | Lower cross-node   | Multi-node training where cross-node bandwidth is limited |

### Configuration

```yaml
model:
  name: "meta-llama/Llama-3.2-3B-Instruct"
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

training:
  mode: "lora"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4

lora:
  r: 16
  alpha: 32

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fsdp_sharding_strategy: "FULL_SHARD"
```

### FSDP Internals

llm-forge generates FSDP configs via `llm_forge.training.distributed.fsdp_config.generate_fsdp_config()`. The generated config is compatible with HuggingFace `TrainingArguments` and the `accelerate` FSDP plugin. Key options:

```python
from llm_forge.training.distributed.fsdp_config import generate_fsdp_config

fsdp_config = generate_fsdp_config(
    config=llm_forge_config,
    sharding_strategy="FULL_SHARD",
    auto_wrap_policy="TRANSFORMER_BASED_WRAP",
    transformer_layer_cls=["LlamaDecoderLayer"],
    backward_prefetch="BACKWARD_PRE",
    forward_prefetch=True,
    cpu_offload=False,
    sync_module_states=True,
    use_orig_params=True,
    limit_all_gathers=True,
    activation_checkpointing=False,
    mixed_precision="bf16",
)
```

The auto-wrap policy is critical: FSDP needs to know which layers to wrap. llm-forge ships defaults for common architectures (LlamaDecoderLayer, MistralDecoderLayer, Phi3DecoderLayer, GPT2Block, and others). If you are using a non-standard architecture, pass the correct layer class name.

### Accelerate Config File

For more control, create an `accelerate_config.yaml`:

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 4
```

Launch:

```bash
accelerate launch --config_file accelerate_config.yaml \
  -m llm_forge.cli train --config config.yaml
```

---

## DeepSpeed ZeRO (Stages 0-3)

DeepSpeed provides progressive memory optimization through its ZeRO (Zero Redundancy Optimizer) stages. Each successive stage shards more training state across GPUs, reducing per-GPU memory at the cost of increased communication.

### When to Use DeepSpeed

- You have GPUs without NVLink (PCIe interconnect).
- You need CPU offloading to train models larger than your total VRAM.
- Your model is 8B+ and you need aggressive memory optimization.
- You want NVMe offloading for extremely large models (ZeRO-Infinity).

### ZeRO Stage Comparison

| Stage | Optimizer States | Gradients | Parameters | Communication | Per-GPU Memory (7B bf16, 4 GPUs) |
|-------|-----------------|-----------|------------|---------------|-----------------------------------|
| 0     | Replicated      | Replicated| Replicated | All-reduce     | ~112 GB (no savings)             |
| 1     | Sharded         | Replicated| Replicated | All-reduce + gather | ~35 GB                     |
| 2     | Sharded         | Sharded   | Replicated | Reduce-scatter + gather | ~21 GB                 |
| 3     | Sharded         | Sharded   | Sharded    | All-gather (forward + backward) | ~8 GB            |

**Stage 0** is equivalent to plain DDP. Use it as a baseline.

**Stage 1** shards only optimizer states. This yields the biggest memory savings per communication cost because optimizer states are by far the largest component (12 bytes/param for AdamW). Minimal communication overhead.

**Stage 2** additionally shards gradients. Good balance between memory and throughput. This is the recommended starting point for most workloads.

**Stage 3** shards everything, including parameters. Each GPU holds only 1/N of the model. Parameters must be all-gathered before every forward and backward pass. Highest communication cost but enables training models that would otherwise not fit at all.

### Configuration

#### ZeRO Stage 2 (Recommended Default)

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
  deepspeed_offload: false

training:
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
```

#### ZeRO Stage 3 (Maximum Sharding)

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 3
  deepspeed_offload: false
```

#### ZeRO-Offload (CPU Offloading)

Offloads optimizer states (Stage 2) or optimizer states + parameters (Stage 3) to CPU RAM. This lets you train larger models at the expense of throughput (CPU-GPU data transfers become a bottleneck).

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
  deepspeed_offload: true    # Offload optimizer states to CPU
```

#### ZeRO-Infinity (NVMe Offloading)

Stage 3 can additionally offload to NVMe SSDs, enabling training of models far exceeding total system memory:

```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "nvme",
      "nvme_path": "/scratch/deepspeed_nvme",
      "pin_memory": true
    },
    "offload_param": {
      "device": "nvme",
      "nvme_path": "/scratch/deepspeed_nvme",
      "pin_memory": true
    }
  }
}
```

### DeepSpeed JSON Config (Advanced)

llm-forge generates DeepSpeed configs programmatically via `generate_deepspeed_config()`:

```python
from llm_forge.training.distributed.deepspeed_config import generate_deepspeed_config

ds_config = generate_deepspeed_config(
    config=llm_forge_config,
    zero_stage=2,
    cpu_offload=False,
    nvme_offload=False,
    gradient_accumulation_steps=4,
    train_micro_batch_size=4,
    gradient_clipping=1.0,
    bf16_enabled=True,
    communication_overlap=True,
    contiguous_gradients=True,
    reduce_bucket_size=500_000_000,
    allgather_bucket_size=500_000_000,
    save_path="ds_config.json",    # Optionally write to disk
)
```

For manual override, you can also provide a standalone JSON file:

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 500000000,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 500000000,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "wall_clock_breakdown": false
}
```

Launch:

```bash
deepspeed --num_gpus=4 \
  -m llm_forge.cli train \
  --config config.yaml \
  --deepspeed ds_config.json
```

### ZeRO Stage 3 + Offload (Maximum Model Size)

For the absolute largest models your hardware can handle:

```json
{
  "bf16": { "enabled": true },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1000000000,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1000000000,
    "stage3_max_reuse_distance": 1000000000,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

---

## Megatron-Core: 3D Parallelism

For models exceeding 20B parameters, or when you have many GPUs across multiple nodes, Megatron-Core provides 3D parallelism by combining three orthogonal parallelism dimensions:

- **Tensor Parallelism (TP)**: Splits each transformer layer's weight matrices across GPUs. Attention heads are divided among TP ranks, and MLP columns/rows are split accordingly. Requires NVLink (very high bandwidth per-operation communication).
- **Pipeline Parallelism (PP)**: Assigns contiguous groups of transformer layers to different GPUs. Uses micro-batching to keep all stages busy (1F1B schedule). Inter-stage communication is smaller (just activations between stages).
- **Sequence Parallelism (SP)**: Partitions the sequence dimension for LayerNorm and dropout operations. Reduces activation memory proportional to the TP degree. Always used when TP is enabled.
- **Data Parallelism (DP)**: Replicates the model shard and distributes data. Combines with TP and PP: `total_gpus = TP * PP * DP`.

### When to Use Megatron

- Model is 20B+ parameters.
- You have 8+ GPUs, ideally across multiple nodes with InfiniBand.
- You need efficient scaling beyond what ZeRO provides.
- You are pre-training (not just fine-tuning).

### Configuration

```yaml
distributed:
  enabled: true
  framework: "megatron"
  num_gpus: 8
  tensor_parallel_degree: 4     # Split each layer across 4 GPUs
  pipeline_parallel_degree: 2   # 2 pipeline stages
  # DP is automatically: 8 / (4 * 2) = 1
```

### Parallelism Dimension Combinations

| Total GPUs | TP | PP | DP | Use Case                                        |
|------------|----|----|----|-------------------------------------------------|
| 8          | 1  | 1  | 8  | Pure data parallel (small model)                |
| 8          | 8  | 1  | 1  | Pure tensor parallel (model fits in 8 GPUs)     |
| 8          | 4  | 1  | 2  | TP + DP (moderate model, want data parallelism) |
| 8          | 4  | 2  | 1  | TP + PP (large model, limited GPUs)             |
| 16         | 4  | 2  | 2  | Full 3D parallelism                             |
| 64         | 8  | 4  | 2  | Large-scale 3D parallelism (multi-node)         |

### Guidelines for Choosing TP and PP

- **TP should not exceed the number of attention heads.** If the model has 32 heads, TP can be 1, 2, 4, 8, 16, or 32.
- **TP should be within a single node** (NVLink bandwidth required). On 8-GPU nodes, TP <= 8.
- **PP is cross-node friendly.** The communication is lighter (just boundary activations). Use PP when you need more parallelism than TP can provide within a node.
- **DP fills the remaining GPUs.** DP = total_gpus / (TP * PP).

### Programmatic Config Generation

```python
from llm_forge.training.distributed.megatron_config import generate_megatron_config

megatron_config = generate_megatron_config(
    config=llm_forge_config,
    tensor_parallel_size=4,
    pipeline_parallel_size=2,
    sequence_parallel=True,
    micro_batch_size=1,
    global_batch_size=64,
    num_layers=40,
    hidden_size=5120,
    num_attention_heads=40,
    num_key_value_heads=8,      # GQA
    seq_length=4096,
    use_flash_attention=True,
    use_rotary_position_embeddings=True,
    normalization="RMSNorm",
    activation_function="swiglu",
    use_distributed_optimizer=True,
    overlap_grad_reduce=True,
    overlap_param_gather=True,
    bf16=True,
    recompute_method="uniform",
    recompute_granularity="selective",
)
```

---

## FP8 Training on Hopper GPUs

NVIDIA Hopper-generation GPUs (H100, H200) support FP8 (8-bit floating point) computation through NVIDIA Transformer Engine, providing up to 2x throughput improvement over bf16.

### Configuration

```yaml
distributed:
  enabled: true
  fp8_enabled: true
  fp8_format: "HYBRID"          # E4M3 for forward, E5M2 for backward

model:
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

training:
  bf16: true
```

### FP8 Formats

| Format   | Forward Pass | Backward Pass | Description                             |
|----------|-------------|---------------|-----------------------------------------|
| `HYBRID` | E4M3        | E5M2          | Recommended: more range for gradients   |
| `E4M3`   | E4M3        | E4M3          | More precision, less dynamic range      |

### Requirements

- NVIDIA Hopper GPU (compute capability >= 9.0)
- CUDA 12.0+
- Transformer Engine: `pip install llm-forge[distributed]`

llm-forge's `auto_optimize_config()` automatically enables FP8 when Hopper GPUs are detected. The `HardwareProfile` checks `GPUInfo.supports_fp8` (compute capability >= 9.0) and sets `fp8_enabled: true` and `fp8_format: "HYBRID"`.

---

## Multi-GPU Setup on a Single Machine

### Basic Setup

```yaml
distributed:
  enabled: true
  framework: "auto"        # Let llm-forge choose
  num_gpus: 4              # Use all 4 GPUs
```

### Launch Commands

```bash
# Using accelerate (recommended -- handles all framework setup)
accelerate launch --num_processes 4 \
  -m llm_forge.cli train --config config.yaml

# Using torchrun (PyTorch native)
torchrun --nproc_per_node=4 \
  -m llm_forge.cli train --config config.yaml

# Using deepspeed launcher
deepspeed --num_gpus=4 \
  -m llm_forge.cli train --config config.yaml
```

### Effective Batch Size

With distributed training, the effective batch size is:

```
effective_batch = per_device_batch_size * gradient_accumulation_steps * num_gpus
```

Example: `per_device=4 * accum=4 * gpus=4 = 64`. When adding GPUs, reduce `gradient_accumulation_steps` proportionally to maintain the same effective batch size.

### GPU Visibility

Control which GPUs are used:

```bash
# Use only GPUs 0 and 1
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
  -m llm_forge.cli train --config config.yaml

# Exclude specific GPUs (useful on shared clusters)
CUDA_VISIBLE_DEVICES=0,1,2,3 ...
```

### NVLink Detection

llm-forge detects NVLink topology via `nvidia-smi topo -m` and uses it for framework selection:

```python
from llm_forge.config.hardware_detector import detect_hardware

profile = detect_hardware()
print(profile.nvlink.has_nvlink)           # True if NVLink detected
print(profile.nvlink.gpu_pairs_connected)  # [(0, 1), (2, 3), ...]
```

When NVLink is present, `auto_optimize_config()` prefers FSDP (lower-overhead sharding). Without NVLink, it selects DeepSpeed (more tolerant of PCIe bandwidth constraints).

---

## Multi-Node Training with SLURM

### Configuration

```yaml
distributed:
  enabled: true
  framework: "deepspeed"     # or "fsdp"
  num_gpus: 8                # GPUs per node
  num_nodes: 2               # Total nodes
  deepspeed_stage: 2
```

### SLURM Job Script (2 Nodes, 8 GPUs Each)

```bash
#!/bin/bash
#SBATCH --job-name=llm-train
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100.80gb:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --exclude=gpu032,dgx003

# Environment setup
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=ib0      # InfiniBand interface

# Master address from SLURM
export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=29500

# Launch with torchrun (one process per node, torchrun spawns per-GPU workers)
srun torchrun \
  --nproc_per_node=8 \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m llm_forge.cli train --config config.yaml
```

### DeepSpeed Multi-Node with SLURM

```bash
#!/bin/bash
#SBATCH --job-name=ds-train
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:A100.80gb:8
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=gpu032,dgx003

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Create hostfile from SLURM node list
scontrol show hostname $SLURM_NODELIST | while read host; do
  echo "$host slots=8"
done > hostfile.txt

deepspeed --hostfile hostfile.txt \
  --num_nodes=$SLURM_NNODES \
  --num_gpus=8 \
  -m llm_forge.cli train \
  --config config.yaml \
  --deepspeed ds_config.json
```

### Networking Considerations

- **InfiniBand** -- preferred for multi-node. Set `NCCL_IB_DISABLE=0` (default). Verify with `ibstat`.
- **RoCE (RDMA over Converged Ethernet)** -- set `NCCL_IB_HCA` to the correct adapter.
- **TCP fallback** -- set `NCCL_SOCKET_IFNAME` to the correct interface (e.g., `eth0`).
- **Firewall** -- ensure port 29500 (or your `MASTER_PORT`) is open between all nodes.
- **NCCL tree algorithms** -- set `NCCL_TREE_THRESHOLD=0` to force tree-based all-reduce (often faster on multi-node).

---

## Framework Selection Decision Guide

```
Is your model larger than your single-GPU VRAM?
|
+-- No --> Use single GPU (no distributed config needed)
|          Use LoRA/QLoRA if memory is tight
|
+-- Yes --> Do you have multiple GPUs?
            |
            +-- No --> Use QLoRA (4-bit) + gradient checkpointing
            |          Or CPU offload: deepspeed_stage: 2, deepspeed_offload: true
            |
            +-- Yes --> Are GPUs on the same node?
                        |
                        +-- Yes --> Is model < 13B?
                        |           |
                        |           +-- Yes --> FSDP FULL_SHARD (with NVLink)
                        |           |           DeepSpeed ZeRO-2 (without NVLink)
                        |           |
                        |           +-- No  --> DeepSpeed ZeRO-3
                        |
                        +-- No (multi-node) --> Is model < 20B?
                                                |
                                                +-- Yes --> DeepSpeed ZeRO-2/3
                                                |           FSDP HYBRID_SHARD
                                                |
                                                +-- No  --> Is model < 70B?
                                                            |
                                                            +-- Yes --> DeepSpeed ZeRO-3 + TP
                                                            |
                                                            +-- No  --> Megatron TP + PP + DP
```

### Performance Comparison (7B LoRA, 4x A100 80 GB)

| Strategy                   | Throughput (relative) | VRAM per GPU | Complexity |
|----------------------------|-----------------------|-------------|------------|
| DDP (no sharding)          | 1.0x (baseline)       | ~18 GB      | Simple     |
| FSDP `FULL_SHARD`          | ~0.9x                 | ~8 GB       | Moderate   |
| DeepSpeed ZeRO-2           | ~0.95x                | ~10 GB      | Moderate   |
| DeepSpeed ZeRO-3           | ~0.8x                 | ~6 GB       | Moderate   |
| ZeRO-3 + CPU offload       | ~0.5x                 | ~4 GB       | Moderate   |

### Memory Usage Estimates (7B Model)

| Configuration                       | Approximate VRAM per GPU |
|-------------------------------------|--------------------------|
| Full bf16, no optimization          | ~112 GB (does not fit)   |
| Full bf16 + gradient checkpointing  | ~45 GB                   |
| LoRA bf16                           | ~16 GB                   |
| QLoRA (4-bit)                       | ~6 GB                    |
| QLoRA + gradient checkpointing      | ~5 GB                    |
| FSDP Full Shard (4 GPUs)            | ~8 GB per GPU            |
| DeepSpeed ZeRO-3 (4 GPUs)           | ~8 GB per GPU            |
| DeepSpeed ZeRO-3 + CPU offload      | ~4 GB per GPU (+ CPU)    |

---

## Example Configs for Each Framework

### 1. FSDP -- 8B Model on 4x A100

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096

data:
  train_path: "HuggingFaceH4/ultrachat_200k"
  format: "sharegpt"

training:
  mode: "lora"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 2
  num_epochs: 1
  learning_rate: 2e-4

lora:
  r: 16
  alpha: 32

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fsdp_sharding_strategy: "FULL_SHARD"
```

### 2. DeepSpeed ZeRO-2 -- 13B Model on 4x A100

```yaml
model:
  name: "meta-llama/Llama-3.2-13B-Instruct"
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096

data:
  train_path: "HuggingFaceH4/ultrachat_200k"
  format: "sharegpt"

training:
  mode: "lora"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  num_epochs: 1
  learning_rate: 1e-4

lora:
  r: 16
  alpha: 32

distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
  deepspeed_offload: false
```

### 3. DeepSpeed ZeRO-3 with CPU Offload -- 70B on 8x A100

```yaml
model:
  name: "meta-llama/Llama-3.1-70B"
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096

data:
  train_path: "HuggingFaceH4/ultrachat_200k"
  format: "sharegpt"

training:
  mode: "qlora"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
  num_epochs: 1
  learning_rate: 5e-5
  optim: "paged_adamw_8bit"

lora:
  r: 16
  alpha: 32

quantization:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true

distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 8
  deepspeed_stage: 3
  deepspeed_offload: true
```

### 4. Megatron-Core -- 70B Pre-training on 64 GPUs (8 nodes)

```yaml
model:
  name: "llama-70b"
  torch_dtype: "bf16"
  max_seq_length: 4096

data:
  train_path: "/data/pretrain_corpus"
  format: "completion"

training:
  mode: "pretrain"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 8
  num_epochs: 1
  learning_rate: 3e-4
  warmup_steps: 2000
  lr_scheduler_type: "cosine"

distributed:
  enabled: true
  framework: "megatron"
  num_gpus: 8                     # Per node
  num_nodes: 8                    # 64 GPUs total
  tensor_parallel_degree: 8       # TP within each node
  pipeline_parallel_degree: 4     # PP across nodes
  # DP = 64 / (8 * 4) = 2
```

### 5. FP8 Training on H100 -- 8B Full Fine-tune

```yaml
model:
  name: "meta-llama/Llama-3.1-8B-Instruct"
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"
  max_seq_length: 4096

data:
  train_path: "HuggingFaceH4/ultrachat_200k"
  format: "sharegpt"

training:
  mode: "full"
  bf16: true
  gradient_checkpointing: true
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 4
  num_epochs: 1

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 8
  fsdp_sharding_strategy: "FULL_SHARD"
  fp8_enabled: true
  fp8_format: "HYBRID"
```

---

## Troubleshooting

### NCCL Communication Errors

```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# If NCCL hangs, try increasing the timeout
export NCCL_TIMEOUT=1800
```

### OOM with FSDP

1. Switch from `SHARD_GRAD_OP` to `FULL_SHARD`.
2. Enable `gradient_checkpointing: true`.
3. Reduce `per_device_train_batch_size` to 1.
4. Increase `gradient_accumulation_steps` proportionally.
5. Reduce `max_seq_length` if possible.
6. If still OOM, switch to DeepSpeed ZeRO-3 with CPU offloading.

### DeepSpeed Hanging at Initialization

1. Ensure all nodes can reach each other on the `MASTER_PORT` (default 29500).
2. Check firewall rules: `nc -zv $MASTER_ADDR $MASTER_PORT`.
3. Set `NCCL_SOCKET_IFNAME` to the correct network interface.
4. Verify that all nodes have the same software environment (PyTorch version, CUDA version, DeepSpeed version).
5. If using SLURM, ensure `--ntasks-per-node` matches `--gres=gpu:N`.

### Slow Multi-Node Training

1. Verify InfiniBand connectivity: `ibstat` should show Active.
2. Enable NCCL tree algorithms: `export NCCL_TREE_THRESHOLD=0`.
3. Use `HYBRID_SHARD` (FSDP) or Stage 2 (DeepSpeed) to minimize cross-node communication.
4. Check that gradient accumulation steps are high enough -- more local steps means less frequent cross-node synchronization.
5. Profile with `NCCL_DEBUG=INFO` to identify bottlenecks (look for `AllReduce` or `AllGather` times).

### Gradient Accumulation Loss Scaling

When using gradient accumulation with distributed training, ensure the loss is averaged correctly. Set `average_tokens_across_devices: true` in your training config (default). Without this, the loss may appear artificially low or high when batch sizes differ across devices due to sequence length variations.

### `auto_optimize_config` Overrides Your Settings

The hardware auto-optimizer aggressively overrides batch size and gradient checkpointing settings for known GPU classes. If you have carefully tuned these values (e.g., on A100 80 GB where it defaults to batch_size=16), use the `--no-auto-optimize` flag:

```bash
llm-forge train --config config.yaml --no-auto-optimize
```

### Mixed Precision Mismatches

- Cannot enable both `bf16: true` and `fp16: true` simultaneously (validated by the schema).
- bf16 requires Ampere+ GPUs (compute capability >= 8.0). On older GPUs, use `fp16: true`.
- FP8 requires Hopper+ GPUs (compute capability >= 9.0). The `auto_optimize_config` guards against this.
- If you see `RuntimeError: expected scalar type BFloat16 but found Float`, ensure your `model.torch_dtype` matches your training precision.

### Checkpoint Saving with FSDP / ZeRO-3

When using FSDP `FULL_SHARD` or ZeRO-3, model parameters are sharded. Saving requires gathering parameters:

- FSDP: uses `SHARDED_STATE_DICT` by default (saves sharded, loads sharded). For a single-file checkpoint, use `FULL_STATE_DICT` (gathers to rank 0).
- DeepSpeed ZeRO-3: set `stage3_gather_16bit_weights_on_model_save: true` in the DeepSpeed config. Without this, saved checkpoints are unusable for single-GPU inference.

---

## Next Steps

- [Training Guide](training_guide.md) -- training modes and hyperparameter tuning
- [Hopper Cluster Guide](hopper_cluster_guide.md) -- GMU Hopper-specific setup and SBATCH scripts
- [Configuration Reference](configuration.md) -- complete schema field reference
- [API Reference](api_reference.md) -- Python API for programmatic use
