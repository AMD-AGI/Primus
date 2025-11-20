from primus.modules.trainer.backend_registry import BackendRegistry
from primus.modules.trainer.megatron.megatron_adapter import MegatronAdapter

# Backend path name
BackendRegistry.register_path_name("megatron", "Megatron-LM")

# Register adapter
BackendRegistry.register_adapter("megatron", MegatronAdapter)
