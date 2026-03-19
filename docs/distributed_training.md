# Distributed Training Guide

Scale your training across multiple GPUs and nodes using llm-forge's distributed training support.

---

## Overview

llm-forge supports several distributed training strategies through its `distributed` configuration section:

| Framework | Config Value | Description | Best For |
|-----------|-------------|-------------|----------|
| Auto | `auto` | Automatically selects the best strategy | Default |
| FSDP | `fsdp` | Fully Sharded Data Parallelism (PyTorch native) | Multi-GPU, large models |
| DeepSpeed | `deepspeed` | ZeRO optimizer stages 0-3 | Multi-GPU, memory optimization |
| Megatron-LM | `megatron` | Tensor + Pipeline parallelism | Very large models, multi-node |

---

## Multi-GPU Training

### Basic Multi-GPU Setup

```yaml
distributed:
  enabled: true
  framework: "auto"
  num_gpus: 4
```

Launch with:

```bash
# Using accelerate (recommended)
accelerate launch --num_processes 4 \
  -m llm_forge.cli train --config config.yaml

# Using torchrun
torchrun --nproc_per_node=4 \
  -m llm_forge.cli train --config config.yaml
```

### Effective Batch Size Calculation

With distributed training, the effective batch size becomes:

```
effective_batch_size = per_device_batch_size * gradient_accumulation_steps * num_gpus
```

Example: `4 * 4 * 4 = 64 effective batch size`

Adjust `gradient_accumulation_steps` when changing `num_gpus` to maintain the same effective batch size.

---

## FSDP (Fully Sharded Data Parallelism)

FSDP is PyTorch's native solution for training models that do not fit in a single GPU. It shards model parameters, gradients, and optimizer states across GPUs.

### Sharding Strategies

| Strategy | Config Value | Description | Memory Savings |
|----------|-------------|-------------|----------------|
| Full Shard | `FULL_SHARD` | Shard everything (params + grads + optimizer) | Maximum |
| Shard Grad Op | `SHARD_GRAD_OP` | Shard gradients and optimizer states only | Moderate |
| No Shard | `NO_SHARD` | Standard DDP (no sharding) | None |
| Hybrid Shard | `HYBRID_SHARD` | Full shard within node, replicate across nodes | Good for multi-node |

### Configuration

```yaml
distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fsdp_sharding_strategy: "FULL_SHARD"

training:
  gradient_checkpointing: true    # Recommended with FSDP
  bf16: true
```

### FSDP with LoRA

FSDP works with LoRA adapters. The base model parameters are sharded across GPUs while LoRA adapter parameters are small enough to be replicated:

```yaml
training:
  mode: "lora"

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fsdp_sharding_strategy: "FULL_SHARD"

lora:
  r: 16
  alpha: 32
```

### FSDP Accelerate Config

Create an `accelerate_config.yaml`:

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

## DeepSpeed ZeRO

DeepSpeed provides progressive memory optimization through its ZeRO (Zero Redundancy Optimizer) stages.

### ZeRO Stages

| Stage | What is Sharded | Memory Reduction | Communication Overhead |
|-------|----------------|-----------------|----------------------|
| Stage 0 | Nothing (DDP) | None | Low |
| Stage 1 | Optimizer states | ~4x optimizer memory | Low |
| Stage 2 | Optimizer states + Gradients | ~8x optimizer memory | Moderate |
| Stage 3 | Everything (params + grads + optimizer) | Linear with GPU count | Higher |

### ZeRO Stage 2 (Recommended Starting Point)

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
```

### ZeRO Stage 3 (Maximum Memory Savings)

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 3
  deepspeed_offload: false
```

### ZeRO-Offload (CPU Offloading)

Offload optimizer states and/or parameters to CPU RAM for even larger models:

```yaml
distributed:
  enabled: true
  framework: "deepspeed"
  num_gpus: 4
  deepspeed_stage: 2
  deepspeed_offload: true     # Offload to CPU
```

### DeepSpeed JSON Config

For advanced DeepSpeed configuration, create a `ds_config.json`:

```json
{
  "bf16": {
    "enabled": true
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
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

### ZeRO Stage 3 with Offload

For the absolute maximum model size on limited hardware:

```json
{
  "bf16": {
    "enabled": true
  },
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
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```

---

## FP8 Training (Hopper GPUs)

FP8 (8-bit floating point) training via NVIDIA Transformer Engine provides up to 2x throughput improvement on Hopper-architecture GPUs (H100, H200).

### Configuration

```yaml
distributed:
  enabled: true
  fp8_enabled: true
  fp8_format: "HYBRID"         # E4M3 for forward, E5M2 for backward

model:
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

training:
  bf16: true
```

### FP8 Formats

| Format | Config Value | Description |
|--------|-------------|-------------|
| Hybrid | `HYBRID` | E4M3 for forward pass, E5M2 for backward pass (recommended) |
| E4M3 | `E4M3` | E4M3 for both forward and backward |

### Requirements

- NVIDIA Hopper GPU (H100 or H200)
- CUDA 12.0+
- Transformer Engine: `pip install llm-forge[distributed]`

---

## Memory Optimization Techniques

### Gradient Checkpointing

Trades compute for memory by recomputing activations during the backward pass instead of storing them:

```yaml
training:
  gradient_checkpointing: true
```

Typical VRAM savings: 30-50% reduction in activation memory.

### Auto Micro-Batch Size

Automatically finds the largest micro-batch size that fits in GPU memory:

```yaml
distributed:
  auto_micro_batch: true
```

### Combined Optimization Strategy

For maximum memory efficiency, combine multiple techniques:

```yaml
model:
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

training:
  mode: "qlora"                          # 4-bit base model
  gradient_checkpointing: true           # Reduce activation memory
  per_device_train_batch_size: 1         # Minimum batch
  gradient_accumulation_steps: 16        # Compensate with accumulation
  optim: "paged_adamw_32bit"             # Paged optimizer

distributed:
  enabled: true
  framework: "deepspeed"
  deepspeed_stage: 2
  num_gpus: 4
```

### Memory Usage Estimates (7B Model)

| Configuration | Approximate VRAM per GPU |
|--------------|--------------------------|
| Full FP16, no optimization | ~28 GB |
| Full FP16 + gradient checkpointing | ~18 GB |
| LoRA FP16 | ~16 GB |
| QLoRA (4-bit) | ~6 GB |
| QLoRA + gradient checkpointing | ~5 GB |
| FSDP Full Shard (4 GPUs) | ~8 GB per GPU |
| DeepSpeed ZeRO-3 (4 GPUs) | ~8 GB per GPU |
| DeepSpeed ZeRO-3 + Offload | ~4 GB per GPU (+ CPU RAM) |

---

## Multi-Node Training

Scale training across multiple machines in a cluster.

### Configuration

```yaml
distributed:
  enabled: true
  framework: "deepspeed"      # or "fsdp"
  num_gpus: 8                  # GPUs per node
  num_nodes: 2                 # Total number of nodes
  deepspeed_stage: 2
```

### Launch with torchrun

On each node, run:

```bash
# Node 0 (master)
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=0 \
  --master_addr=NODE0_IP \
  --master_port=29500 \
  -m llm_forge.cli train --config config.yaml

# Node 1
torchrun \
  --nproc_per_node=8 \
  --nnodes=2 \
  --node_rank=1 \
  --master_addr=NODE0_IP \
  --master_port=29500 \
  -m llm_forge.cli train --config config.yaml
```

### Launch with DeepSpeed

```bash
deepspeed \
  --num_nodes=2 \
  --num_gpus=8 \
  --hostfile hostfile.txt \
  -m llm_forge.cli train --config config.yaml
```

Where `hostfile.txt` contains:

```
node0 slots=8
node1 slots=8
```

### Networking Considerations

- Use InfiniBand or RoCE for inter-node communication when available
- Ensure NCCL can discover all GPUs across nodes
- Set `NCCL_IB_DISABLE=0` to enable InfiniBand
- Set `NCCL_SOCKET_IFNAME` to the correct network interface

---

## Tensor and Pipeline Parallelism (Megatron-LM)

For very large models (13B+), tensor and pipeline parallelism split individual layers across GPUs.

### Configuration

```yaml
distributed:
  enabled: true
  framework: "megatron"
  num_gpus: 8
  tensor_parallel_degree: 4      # Split each layer across 4 GPUs
  pipeline_parallel_degree: 2    # Pipeline across 2 stages
```

### Parallelism Combinations

| Strategy | GPUs | TP Degree | PP Degree | DP Degree |
|----------|------|-----------|-----------|-----------|
| Pure Data Parallel | 8 | 1 | 1 | 8 |
| Pure Tensor Parallel | 8 | 8 | 1 | 1 |
| TP + DP | 8 | 4 | 1 | 2 |
| TP + PP | 8 | 4 | 2 | 1 |
| TP + PP + DP | 16 | 4 | 2 | 2 |

---

## Choosing a Strategy

### Decision Guide

```
Do you have multiple GPUs?
├── No -> Use single GPU with QLoRA + gradient checkpointing
├── Yes, same node
│   ├── Model fits in one GPU? -> Standard DDP / FSDP NO_SHARD
│   ├── Model nearly fits? -> FSDP SHARD_GRAD_OP or DeepSpeed ZeRO-2
│   └── Model does not fit? -> FSDP FULL_SHARD or DeepSpeed ZeRO-3
└── Yes, multiple nodes
    ├── Moderate model size -> DeepSpeed ZeRO-2/3
    ├── Large model (13B+) -> Megatron TP + DP
    └── Very large model (70B+) -> Megatron TP + PP + DP
```

### Performance Comparison (7B LoRA, 4x A100 80GB)

| Strategy | Throughput | VRAM per GPU | Setup Complexity |
|----------|-----------|-------------|-----------------|
| DDP (no sharding) | Baseline | ~18 GB | Simple |
| FSDP FULL_SHARD | ~0.9x baseline | ~8 GB | Moderate |
| DeepSpeed ZeRO-2 | ~0.95x baseline | ~10 GB | Moderate |
| DeepSpeed ZeRO-3 | ~0.8x baseline | ~6 GB | Moderate |
| ZeRO-3 + Offload | ~0.5x baseline | ~4 GB | Moderate |

---

## Troubleshooting

### NCCL Communication Errors

```bash
# Check NCCL version
python -c "import torch; print(torch.cuda.nccl.version())"

# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
```

### OOM with FSDP

1. Switch from `SHARD_GRAD_OP` to `FULL_SHARD`
2. Enable gradient checkpointing
3. Reduce `per_device_train_batch_size` to 1
4. Increase `gradient_accumulation_steps` to compensate

### DeepSpeed Hanging at Initialization

1. Ensure all nodes can reach each other on the specified port
2. Check firewall rules for port 29500 (default)
3. Set `export NCCL_SOCKET_IFNAME=eth0` (or your network interface)

### Slow Multi-Node Training

1. Verify InfiniBand connectivity: `ibstat`
2. Enable NCCL tree algorithms: `export NCCL_TREE_THRESHOLD=0`
3. Use `HYBRID_SHARD` for FSDP to minimize cross-node communication

---

## Next Steps

- [Training Guide](training_guide.md) -- training modes and hyperparameters
- [Hopper Cluster Guide](hopper_cluster_guide.md) -- GMU Hopper cluster specifics
- [Configuration Reference](configuration.md) -- distributed config fields
