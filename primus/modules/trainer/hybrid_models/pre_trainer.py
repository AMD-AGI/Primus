###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
import sys
from typing import Any, Dict, Optional

from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule


class HybridModelsPretrainTrainer(BaseModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.primus_cfg = kwargs.pop("primus_config", None)
        if self.primus_cfg is None:
            raise ValueError("primus_config is required")

        pre_trainer_cfg = self.primus_cfg.get_module_config("pre_trainer")
        self.module_config = pre_trainer_cfg
        
        # Setup environment
        self.setup_environment()
        
        # Build training command
        self.training_cmd = self.build_training_command()
        self.log_config()

    def setup_environment(self):
        """Setup environment variables and paths for hybrid models training."""
        backend_path = getattr(self.module_config, 'backend_path', None)
        if backend_path:
            # Add backend path to Python path
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            # Store backend path for later use
            self.backend_path = backend_path
        else:
            self.backend_path = '.'
            
        # Set environment variables
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            os.environ[key] = value

    def build_training_command(self):
        """Build the training command using the existing Primus run_local_pretrain.sh script."""
        # Get the EXP config path from training_command parameters
        training_cmd = getattr(self.module_config, 'training_command', {})
        exp_config = training_cmd.get('model_config', 'configs/llama3.2_1B/zebra_4MLA12M2_8bt_SFT.yaml')
        
        # Set up the run script path (use the Primus examples script)
        primus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        run_script_path = os.path.join(primus_root, 'examples', 'run_local_pretrain.sh')
        
        # Use bash to execute the script with EXP environment variable
        cmd = ['bash', run_script_path]
        
        # Store the EXP value for later use
        self.exp_config_path = exp_config
        
        return cmd

    def log_config(self):
        """Log the training configuration."""
        from primus.core.utils.logger import _logger as primus_logger
        
        primus_logger.info("========== Hybrid Models Training Config ==========")
        primus_logger.info(f"Backend Path: {self.backend_path}")
        
        # Get the actual script path
        primus_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        run_script_path = os.path.join(primus_root, 'examples', 'run_local_pretrain.sh')
        
        primus_logger.info(f"Run Script: {run_script_path}")
        primus_logger.info(f"EXP Config: {self.exp_config_path}")
        primus_logger.info(f"Training Command: EXP={self.exp_config_path} bash ./examples/run_local_pretrain.sh")
        
        # Log environment variables
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            primus_logger.info(f"  env.{key}: {value}")

    def setup(self):
        """Setup phase - change to backend directory."""
        if hasattr(self, 'backend_path'):
            original_cwd = os.getcwd()
            os.chdir(self.backend_path)
            from primus.core.utils.logger import _logger as primus_logger
            primus_logger.info(f"Changed working directory to: {self.backend_path}")
            self.original_cwd = original_cwd

    def init(self, *init_args, **kwargs):
        """Initialize the trainer - no special initialization needed for hybrid models."""
        pass

    def run(self, *args, **kwargs):
        """Run the hybrid models training."""
        from primus.core.utils.logger import _logger as primus_logger
        
        primus_logger.info(f"Starting hybrid models training...")
        primus_logger.info(f"Command: EXP={self.exp_config_path} bash ./examples/run_local_pretrain.sh")
        primus_logger.info(f"Working directory: {os.getcwd()}")
        
        # Set environment variables
        env = os.environ.copy()
        env['EXP'] = self.exp_config_path
        
        # Add any additional environment variables from config
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            env[key] = value
        
        # Execute training
        try:
            result = subprocess.run(
                self.training_cmd, 
                env=env, 
                check=True, 
                capture_output=False
            )
            primus_logger.info(f"Training completed successfully with return code: {result.returncode}")
            return result.returncode
        except subprocess.CalledProcessError as e:
            primus_logger.error(f"Training failed with return code: {e.returncode}")
            raise e
        except Exception as e:
            primus_logger.error(f"Training failed with error: {e}")
            raise e
        finally:
            # Restore original working directory
            if hasattr(self, 'original_cwd'):
                os.chdir(self.original_cwd)
