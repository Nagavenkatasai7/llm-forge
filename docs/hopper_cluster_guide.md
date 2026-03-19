# GMU Hopper Cluster Guide

Run llm-forge training jobs on George Mason University's Hopper HPC cluster using SLURM and Singularity containers.

---

## Cluster Overview

The GMU Hopper cluster is a shared High-Performance Computing (HPC) resource managed by the Office of Research Computing. Key specifications:

| Component | Details |
|-----------|---------|
| Job Scheduler | SLURM |
| Container Runtime | Singularity / Apptainer |
| GPU Nodes | NVIDIA A100 (40GB and 80GB), H100 (80GB) |
| Interconnect | InfiniBand HDR |
| Storage | Lustre parallel filesystem |
| Module System | Lmod |

---

## Initial Setup

### 1. Connect to the Cluster

```bash
ssh your-username@hopper.orc.gmu.edu
```

### 2. Load Required Modules

```bash
module load cuda/12.2
module load singularity/3.11
module load anaconda3
```

Add these to your `~/.bashrc` for automatic loading:

```bash
echo 'module load cuda/12.2 singularity/3.11 anaconda3' >> ~/.bashrc
```

### 3. Create a Conda Environment

```bash
conda create -n llm-forge python=3.11 -y
conda activate llm-forge
pip install llm-forge[all]
```

### 4. Set Up HuggingFace Cache

The home directory on Hopper has limited space. Point the HuggingFace cache to the scratch filesystem:

```bash
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache
mkdir -p $HF_HOME
```

Add to `~/.bashrc`:

```bash
echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
echo 'export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache' >> ~/.bashrc
```

### 5. Authenticate with HuggingFace

```bash
pip install huggingface_hub
huggingface-cli login
```

---

## SLURM Job Submission

### Single-GPU LoRA Training

Create a job script `train_lora.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your-email@gmu.edu

# Load modules
module load cuda/12.2 anaconda3

# Activate environment
conda activate llm-forge

# Set cache directories
export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

# Create output directory
mkdir -p logs

# Run training
llm-forge train --config /scratch/$USER/projects/my-project/config.yaml
```

Submit:

```bash
mkdir -p logs
sbatch train_lora.slurm
```

### Multi-GPU Training (Single Node)

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-multi-gpu
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00

module load cuda/12.2 anaconda3
conda activate llm-forge

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

# Multi-GPU with torchrun
torchrun --nproc_per_node=4 \
  -m llm_forge.cli train \
  --config /scratch/$USER/projects/my-project/config.yaml
```

### Multi-Node Training

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-multinode
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --exclusive

module load cuda/12.2 anaconda3
conda activate llm-forge

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

# Get master node info
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * 4))

# Launch on each node
srun torchrun \
  --nproc_per_node=4 \
  --nnodes=$SLURM_NNODES \
  --node_rank=$SLURM_NODEID \
  --master_addr=$MASTER_ADDR \
  --master_port=$MASTER_PORT \
  -m llm_forge.cli train \
  --config /scratch/$USER/projects/my-project/config.yaml
```

### Multi-Node with DeepSpeed

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-deepspeed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=12:00:00
#SBATCH --exclusive

module load cuda/12.2 anaconda3
conda activate llm-forge

export HF_HOME=/scratch/$USER/hf_cache
export TRANSFORMERS_CACHE=/scratch/$USER/hf_cache

# Generate hostfile for DeepSpeed
scontrol show hostnames $SLURM_JOB_NODELIST > /tmp/hostfile_$SLURM_JOB_ID
sed -i 's/$/ slots=4/' /tmp/hostfile_$SLURM_JOB_ID

# Launch with DeepSpeed
deepspeed \
  --hostfile /tmp/hostfile_$SLURM_JOB_ID \
  --num_nodes=$SLURM_NNODES \
  --num_gpus=4 \
  -m llm_forge.cli train \
  --config /scratch/$USER/projects/my-project/config.yaml

# Cleanup
rm /tmp/hostfile_$SLURM_JOB_ID
```

---

## Singularity / Apptainer Containers

For reproducible environments, use Singularity containers.

### Building a Container

Create a definition file `llm-forge.def`:

```singularity
Bootstrap: docker
From: nvidia/cuda:12.2.2-devel-ubuntu22.04

%post
    apt-get update && apt-get install -y \
        python3.11 python3.11-venv python3-pip git wget \
        && rm -rf /var/lib/apt/lists/*

    python3.11 -m pip install --upgrade pip
    python3.11 -m pip install llm-forge[all]
    python3.11 -m pip install flash-attn --no-build-isolation

%environment
    export LC_ALL=C
    export PATH=/usr/local/bin:$PATH

%runscript
    exec python3.11 -m llm_forge.cli "$@"
```

Build the container:

```bash
# On a build node (not the login node)
singularity build llm-forge.sif llm-forge.def
```

### Running with Singularity

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-singularity
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

module load cuda/12.2 singularity/3.11

export HF_HOME=/scratch/$USER/hf_cache

singularity exec --nv \
  --bind /scratch/$USER:/scratch/$USER \
  --bind $HF_HOME:$HF_HOME \
  /scratch/$USER/containers/llm-forge.sif \
  llm-forge train --config /scratch/$USER/projects/my-project/config.yaml
```

Key flags:
- `--nv` enables NVIDIA GPU support inside the container
- `--bind` mounts host directories into the container

### Multi-GPU with Singularity

```bash
#!/bin/bash
#SBATCH --job-name=llm-forge-singularity-multi
#SBATCH --partition=gpuq
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=08:00:00

module load cuda/12.2 singularity/3.11

export HF_HOME=/scratch/$USER/hf_cache

singularity exec --nv \
  --bind /scratch/$USER:/scratch/$USER \
  --bind $HF_HOME:$HF_HOME \
  /scratch/$USER/containers/llm-forge.sif \
  torchrun --nproc_per_node=4 \
    -m llm_forge.cli train \
    --config /scratch/$USER/projects/my-project/config.yaml
```

---

## GPU Partition Guide

### Available Partitions

| Partition | GPU Type | VRAM | Max Time | Notes |
|-----------|----------|------|----------|-------|
| `gpuq` | A100 40GB/80GB | 40-80 GB | 24h | General GPU queue |
| `contrib-gpuq` | Varies | Varies | 48h | Contributed GPU nodes |

Check available GPUs:

```bash
sinfo -p gpuq -o "%N %G %T %C"
```

### Requesting Specific GPUs

```bash
# Request A100 80GB specifically
#SBATCH --gres=gpu:A100_80:1

# Request any A100
#SBATCH --gres=gpu:A100:1

# Request H100
#SBATCH --gres=gpu:H100:1
```

---

## Configuration for Hopper

### Recommended Config for Single A100 80GB

```yaml
model:
  name: "meta-llama/Llama-3.2-3B"
  max_seq_length: 4096
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  mode: "lora"
  output_dir: "/scratch/$USER/outputs/llama-lora"
  num_epochs: 3
  per_device_train_batch_size: 8
  gradient_accumulation_steps: 2
  learning_rate: 2.0e-4
  bf16: true
  gradient_checkpointing: true

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
  test_size: 0.05
```

### Recommended Config for 4x A100 80GB

```yaml
model:
  name: "meta-llama/Llama-3.1-8B"
  max_seq_length: 4096
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

lora:
  r: 32
  alpha: 64
  target_modules:
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj

training:
  mode: "lora"
  output_dir: "/scratch/$USER/outputs/llama8b-lora"
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-4
  bf16: true
  gradient_checkpointing: true

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fsdp_sharding_strategy: "FULL_SHARD"

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
```

### Full Fine-Tune on H100 with FP8

```yaml
model:
  name: "meta-llama/Llama-3.2-3B"
  max_seq_length: 4096
  torch_dtype: "bf16"
  attn_implementation: "flash_attention_2"

training:
  mode: "full"
  output_dir: "/scratch/$USER/outputs/llama-full"
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2.0e-5
  bf16: true
  gradient_checkpointing: true

distributed:
  enabled: true
  framework: "fsdp"
  num_gpus: 4
  fp8_enabled: true
  fp8_format: "HYBRID"

data:
  train_path: "tatsu-lab/alpaca"
  format: "alpaca"
```

---

## File System Best Practices

### Storage Locations

| Path | Quota | Speed | Persistence | Use For |
|------|-------|-------|-------------|---------|
| `$HOME` | Limited (~50GB) | Moderate | Permanent | Scripts, configs, small files |
| `/scratch/$USER` | Large (~1TB) | Fast (Lustre) | Purged after 90 days | Model weights, datasets, outputs |
| `/tmp` (node-local) | Varies | Very fast | Job lifetime | Temporary cache |

### Recommendations

1. **Store datasets on scratch:** `/scratch/$USER/datasets/`
2. **Store model outputs on scratch:** `/scratch/$USER/outputs/`
3. **Point HF cache to scratch:** `export HF_HOME=/scratch/$USER/hf_cache`
4. **Keep configs in home:** `$HOME/projects/configs/`
5. **Pre-download models** before submitting jobs to avoid wasting GPU hours on downloads:

```bash
# Pre-download on login node
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B'); AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.2-1B')"
```

---

## Monitoring Jobs

### SLURM Commands

```bash
# Check job status
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# View job output in real-time
tail -f logs/llm-forge-lora_<JOB_ID>.out

# Cancel a job
scancel <JOB_ID>

# Check GPU usage on a running job
srun --jobid=<JOB_ID> --pty nvidia-smi

# View past job efficiency
sacct -j <JOB_ID> --format=JobID,Elapsed,MaxRSS,MaxVMSize,AllocCPUS,AllocGRES
```

### WandB Monitoring

Enable WandB for remote monitoring of training progress:

```yaml
training:
  report_to: ["wandb"]
```

Set your API key in the job script:

```bash
export WANDB_API_KEY="your-wandb-api-key"
```

---

## Tips for Hopper

1. **Use interactive sessions for debugging:**

```bash
srun --partition=gpuq --gres=gpu:A100:1 --cpus-per-task=8 --mem=64G --time=01:00:00 --pty bash
```

2. **Pre-download everything** before submitting batch jobs. Network access from compute nodes may be restricted.

3. **Use `--exclusive`** for multi-node jobs to avoid interference from other users' processes.

4. **Set `--time` conservatively** but not too tight. SLURM kills jobs at the time limit with no checkpoint save.

5. **Enable checkpoint saving** so you can resume if a job is preempted:

```yaml
training:
  save_steps: 500
  save_total_limit: 3
  resume_from_checkpoint: null    # Set to checkpoint path when resuming
```

6. **Request the right resources.** Over-requesting GPUs wastes queue priority. Under-requesting causes OOM errors.

7. **Check module compatibility:**

```bash
module avail cuda
module avail singularity
```

8. **Use `sacct` to review completed jobs** and optimize resource requests for future runs.

9. **Store outputs on Lustre `/scratch`** for fast I/O during training. Copy final models to permanent storage after training completes.

10. **Set up email notifications** with `#SBATCH --mail-type=END,FAIL` to know when jobs finish or fail without polling `squeue`.

---

## Next Steps

- [Distributed Training](distributed_training.md) -- general distributed training guide
- [Training Guide](training_guide.md) -- training modes and hyperparameters
- [Configuration Reference](configuration.md) -- full YAML reference
