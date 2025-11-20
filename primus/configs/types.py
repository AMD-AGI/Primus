# primus/config/types.py

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class WorkflowStepConfig:
    """一个要执行的模块（例如 pretrain/sft/eval）"""

    name: str  # 用户自定义名称
    module: str  # pre_trainer / sft_trainer / eval ...
    framework: str  # megatron / torchtitan / maxtext ...
    model: Optional[Union[str, Dict]]  # model 名称或其配置
    overrides: Dict[str, Any]  # 覆盖 backend 默认值
    raw_cfg: Dict[str, Any]  # 原始 YAML，用于 debug


@dataclass
class PrimusConfig:
    """整个 Primus 配置（对应完整 YAML）"""

    work_group: str
    user_name: str
    exp_name: str
    workspace: str
    modules: List[WorkflowStepConfig]
