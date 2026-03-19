"""Distributed training framework selection and configuration.

Provides the :class:`DistributedOrchestrator` which selects the optimal
distributed framework (FSDP, DeepSpeed, or Megatron-Core) and generates
the corresponding configuration.
"""

from llm_forge.training.distributed.orchestrator import DistributedOrchestrator

__all__ = [
    "DistributedOrchestrator",
]
