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
        # First try to get backend_path from module_config
        backend_path = getattr(self.module_config, 'backend_path', None)
        
        # If not found in module_config, try to get it from the primus_cfg
        if not backend_path:
            backend_path = getattr(self.primus_cfg, 'backend_path', None)
        
        # If still not found, try to get it from the root config
        if not backend_path:
            backend_path = getattr(self.primus_cfg, 'modules', {}).get('pre_trainer', {}).get('backend_path', None)
        
        if backend_path:
            # Add backend path to Python path
            if backend_path not in sys.path:
                sys.path.insert(0, backend_path)
            
            # Store backend path for later use
            self.backend_path = backend_path
            
            # Run install.sh in the backend_path directory
            self.run_backend_install()
        else:
            self.backend_path = '.'
            
        # Set environment variables
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            os.environ[key] = value

    def run_backend_install(self):
        """Run the install.sh script in the backend_path directory."""
        from primus.core.utils.logger import _logger as primus_logger
        
        install_script_path = os.path.join(self.backend_path, 'install.sh')
        
        if os.path.exists(install_script_path):
            primus_logger.info(f"Running install.sh in backend directory: {self.backend_path}")
            primus_logger.info("Command: bash install.sh FLASH_ATTN=1 MAMBA=1")
            
            try:
                # Change to backend directory and run install script
                original_cwd = os.getcwd()
                os.chdir(self.backend_path)
                
                result = subprocess.run(
                    ['bash', 'install.sh', 'FLASH_ATTN=1', 'MAMBA=1'],
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                primus_logger.info(f"Install script completed successfully")
                if result.stdout:
                    primus_logger.info(f"Install output: {result.stdout}")
                    
            except subprocess.CalledProcessError as e:
                primus_logger.error(f"Install script failed with return code: {e.returncode}")
                if e.stderr:
                    primus_logger.error(f"Install error: {e.stderr}")
                raise e
            except Exception as e:
                primus_logger.error(f"Failed to run install script: {e}")
                raise e
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
        else:
            primus_logger.warning(f"Install script not found at: {install_script_path}")
            primus_logger.warning("Skipping backend installation")

    def build_training_command(self):
        """Build the accelerate launch command using the configuration from the YAML file."""
        # Get the training command parameters from the config
        training_cmd = getattr(self.module_config, 'training_command', {})
        
        # Get the script, accelerate config, and model config from the YAML
        script = training_cmd.get('script', 'train_hybrid/train_distill.py')
        accelerate_config = training_cmd.get('accelerate_config', 'configs/fsdp.yaml')
        model_config = training_cmd.get('model_config', 'configs/llama3.2_1B/zebra_4MLA12M2_8bt_SFT.yaml')
        
        # Get environment variables
        env_vars = getattr(self.module_config, 'env_vars', {})
        accelerate_log_level = env_vars.get('ACCELERATE_LOG_LEVEL', 'info')
        
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
        
        # Store values for later use
        self.script = script
        self.accelerate_config = accelerate_config
        self.model_config = model_config
        self.accelerate_log_level = accelerate_log_level

        from primus.core.utils.logger import _logger as primus_logger
        
        primus_logger.info(f"Script: {self.script}")
        primus_logger.info(f"Accelerate Config: {self.accelerate_config}")
        primus_logger.info(f"Model Config: {self.model_config}")
        primus_logger.info(f"Accelerate Log Level: {self.accelerate_log_level}")
        primus_logger.info(f"Command: {cmd}")
        
        return cmd

    def log_config(self):
        """Log the training configuration."""
        from primus.core.utils.logger import _logger as primus_logger
        
        primus_logger.info("========== Hybrid Models Training Config ==========")
        primus_logger.info(f"Backend Path: {self.backend_path}")
        primus_logger.info(f"Script: {getattr(self, 'script', 'N/A')}")
        primus_logger.info(f"Accelerate Config: {getattr(self, 'accelerate_config', 'N/A')}")
        primus_logger.info(f"Model Config: {getattr(self, 'model_config', 'N/A')}")
        primus_logger.info(f"Accelerate Log Level: {getattr(self, 'accelerate_log_level', 'N/A')}")
        primus_logger.info(f"Training Command: {self.training_cmd}")
        
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
        primus_logger.info(f"Command: {self.training_cmd}")
        primus_logger.info(f"Backend directory: {self.backend_path}")
        
        # Change to backend directory for training
        original_cwd = os.getcwd()
        os.chdir(self.backend_path)
        primus_logger.info(f"Changed working directory to: {os.getcwd()}")
        
        # Debug: Check if config files exist
        accelerate_config_path = getattr(self, 'accelerate_config', 'configs/fsdp.yaml')
        model_config_path = getattr(self, 'model_config', 'configs/llama3.2_1B/zebra_8MLA8M2_8bt_SFT.yaml')
        
        primus_logger.info(f"Checking for accelerate config: {accelerate_config_path}")
        primus_logger.info(f"File exists: {os.path.exists(accelerate_config_path)}")
        primus_logger.info(f"Checking for model config: {model_config_path}")
        primus_logger.info(f"File exists: {os.path.exists(model_config_path)}")
        
        # List files in current directory
        primus_logger.info(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('configs'):
            primus_logger.info(f"Files in configs directory: {os.listdir('configs')}")
        
        # Set environment variables
        env = os.environ.copy()
        
        # Add any additional environment variables from config
        env_vars = getattr(self.module_config, 'env_vars', {})
        for key, value in env_vars.items():
            env[key] = value
        
        # Execute training using shell=True to run the accelerate command
        try:
            # If files don't exist with relative paths, try absolute paths
            if not os.path.exists(accelerate_config_path):
                primus_logger.warning(f"Config file not found with relative path, trying absolute path")
                # Update the command with absolute paths
                abs_accelerate_config = os.path.abspath(accelerate_config_path)
                abs_model_config = os.path.abspath(model_config_path)
                
                if os.path.exists(abs_accelerate_config):
                    primus_logger.info(f"Found config with absolute path: {abs_accelerate_config}")
                    # Rebuild command with absolute paths
                    cmd_parts = [
                        f'ACCELERATE_LOG_LEVEL={self.accelerate_log_level}',
                        'accelerate',
                        'launch',
                        f'--config_file',
                        abs_accelerate_config,
                        self.script,
                        abs_model_config
                    ]
                    self.training_cmd = ' '.join(cmd_parts)
                    primus_logger.info(f"Updated command: {self.training_cmd}")
            
            result = subprocess.run(
                self.training_cmd, 
                shell=True,
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
            os.chdir(original_cwd)
            primus_logger.info(f"Working directory: {os.getcwd()}")
