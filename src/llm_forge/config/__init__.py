"""llm-forge configuration system.

Public API
----------
- :class:`LLMForgeConfig` -- master Pydantic v2 config model
- :func:`validate_config_file` -- load and validate a YAML config file
- :func:`load_preset` -- load a built-in preset by name
- :func:`list_presets` -- list available preset names
- :func:`detect_hardware` -- detect GPUs, RAM, and other hardware
- :func:`auto_optimize_config` -- auto-tune config for detected hardware
- :class:`HardwareProfile` -- dataclass with detected hardware info
"""

from llm_forge.config.hardware_detector import (
    HardwareProfile,
    auto_optimize_config,
    detect_hardware,
)
from llm_forge.config.schema import (
    CloudGPUConfig,
    ComputeBackend,
    ComputeConfig,
    DataCleaningConfig,
    DataConfig,
    DataFormat,
    DeduplicationTier,
    DistributedConfig,
    EvalConfig,
    LLMForgeConfig,
    LoRAConfig,
    ModelConfig,
    PrecisionMode,
    QualityPreset,
    QuantizationConfig,
    RAGConfig,
    ServingConfig,
    SLURMConfig,
    SSHConfig,
    TrainingConfig,
    TrainingMode,
)
from llm_forge.config.validator import (
    ConfigValidationError,
    list_presets,
    load_preset,
    validate_config_dict,
    validate_config_file,
)

__all__ = [
    # Schema
    "LLMForgeConfig",
    "ModelConfig",
    "LoRAConfig",
    "QuantizationConfig",
    "DataCleaningConfig",
    "DataConfig",
    "TrainingConfig",
    "DistributedConfig",
    "EvalConfig",
    "RAGConfig",
    "ServingConfig",
    "ComputeConfig",
    "ComputeBackend",
    "SLURMConfig",
    "SSHConfig",
    "CloudGPUConfig",
    # Enums
    "TrainingMode",
    "DataFormat",
    "PrecisionMode",
    "DeduplicationTier",
    "QualityPreset",
    # Validator
    "validate_config_file",
    "validate_config_dict",
    "load_preset",
    "list_presets",
    "ConfigValidationError",
    # Hardware
    "detect_hardware",
    "auto_optimize_config",
    "HardwareProfile",
]
