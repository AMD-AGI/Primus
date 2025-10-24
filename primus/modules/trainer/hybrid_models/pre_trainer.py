###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import os
import subprocess
import sys
import yaml
from typing import Any, Dict, Optional

from primus.core.utils.yaml_utils import nested_namespace_to_dict
from primus.modules.base_module import BaseModule


class HybridModelsPretrainTrainer(BaseModule):
    def init(self, *args, **kwargs):
        # Get the backend path from the first element of sys.path        
        self.backend_path = sys.path[0] if sys.path else None
        
        if self.backend_path is None:
            raise FileNotFoundError("No backend path found in sys.path")
                
        # Build training command
        self.training_cmd = self.build_training_command()

    def build_training_command(self):
        """Build the accelerate launch command using the configuration from the YAML file."""
        from primus.core.utils.logger import _logger as primus_logger
        # Get the experiment YAML file path from the primus config
        # primus_logger.info(f'Primus config: {self.primus_config}')
        exp_yaml_path = self.primus_config._exp.config_file
        primus_logger.info(f'Experiment YAML file: {exp_yaml_path}')
        
        with open(exp_yaml_path, 'r') as f:
            yaml_content = f.read()        
        yaml_data = yaml.safe_load(yaml_content)
        
        # Extract training_command from the pre_trainer section
        pre_trainer_section = yaml_data.get('modules', {}).get('pre_trainer', {})
        training_cmd = pre_trainer_section.get('training_command', {})
        if not training_cmd:
            raise ValueError("training_command not found in pre_trainer section of YAML")
        
        # Extract variables from the YAML config
        script = training_cmd.get('script', None)
        accelerate_config = training_cmd.get('accelerate_config', None)
        model_config = training_cmd.get('model_config', None)
        accelerate_log_level = training_cmd.get('ACCELERATE_LOG_LEVEL', None)
        
        # Validate that required parameters are present
        if script is None:
            raise ValueError("script not found in training_command config")
        if accelerate_config is None:
            raise ValueError("accelerate_config not found in training_command config")
        if model_config is None:
            raise ValueError("model_config not found in training_command config")
        if accelerate_log_level is None:
            accelerate_log_level = 'info'  # Default fallback
        
        print(f'Training command config: {training_cmd}')
        print(f'Script: {script}')
        print(f'Accelerate config: {accelerate_config}')
        print(f'Model config: {model_config}')
        print(f'Log level: {accelerate_log_level}')
        
        # Construct the accelerate launch command
        cmd_parts = [
            f'ACCELERATE_LOG_LEVEL={accelerate_log_level}',
            'accelerate',
            'launch',
            f'--config_file',
            accelerate_config,
            script,
            model_config
        ]
        
        # Join the command parts
        cmd = ' '.join(cmd_parts)
        
        return cmd

    def run(self, *args, **kwargs):
        """Run the hybrid models training."""
        from primus.core.utils.logger import _logger as primus_logger
        
        primus_logger.info(f"Starting hybrid models training...")
        primus_logger.info(f"Command: {self.training_cmd}")
        primus_logger.info(f"Backend directory: {self.backend_path}")
        
        # Change to backend directory for training
        original_cwd = os.getcwd()
        os.chdir(self.backend_path)
        primus_logger.info(f"Changed working directory to: {os.getcwd()}")
        
        # List files in current directory
        primus_logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Set environment variables
        env = os.environ.copy()
        
        # Add any additional environment variables from config
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            env[key] = value
        
        # Execute training using shell=True to run the accelerate command
        try:
            current_directory = os.getcwd()
            primus_logger.info(f"Current directory: {current_directory}")
            setup_cmd = ["bash", "install.sh", "FLASH_ATTN=1", "MAMBA=1"]
            setup_result = subprocess.Popen(setup_cmd, stdout=sys.stdout, stderr=sys.stderr)
            setup_result.wait()
            primus_logger.info(f"Install script completed successfully with return code: {setup_result.returncode}")

            primus_logger.info(f"Current directory: {current_directory}")
            training_result = subprocess.Popen(self.training_cmd, stdout=sys.stdout, stderr=sys.stderr)
            training_result.wait()
            # training_result = subprocess.run(
            #     self.training_cmd, 
            #     shell=True,
            #     env=env, 
            #     check=True, 
            #     capture_output=True,
            #     text=True
            # )
            primus_logger.info(f"Training completed successfully with return code: {training_result.returncode}")
            return training_result.returncode
        except subprocess.CalledProcessError as e:
            primus_logger.error(f"Training failed with return code: {e.returncode}")
            raise e
        except Exception as e:
            primus_logger.error(f"Training failed with error: {e}")
            raise e
        finally:
            os.chdir(original_cwd)
            primus_logger.info(f"Working directory: {os.getcwd()}")
