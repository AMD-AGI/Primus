###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for MegatronSFTTrainer registration and basic functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

from primus.backends.megatron.megatron_sft_trainer import MegatronSFTTrainer
from primus.core.backend.backend_registry import BackendRegistry


class TestMegatronSFTTrainer(unittest.TestCase):
    """Test MegatronSFTTrainer basic functionality."""

    def test_sft_trainer_registered(self):
        """Test that MegatronSFTTrainer is properly registered."""
        # Check that megatron_sft trainer is registered
        self.assertTrue(
            BackendRegistry.has_trainer_class("megatron_sft"),
            "MegatronSFTTrainer should be registered as 'megatron_sft'"
        )
        
        # Get the trainer class
        trainer_cls = BackendRegistry.get_trainer_class("megatron_sft")
        self.assertEqual(
            trainer_cls.__name__, 
            "MegatronSFTTrainer",
            "Registered trainer should be MegatronSFTTrainer class"
        )

    def test_sft_trainer_inherits_from_base(self):
        """Test that MegatronSFTTrainer inherits from MegatronBaseTrainer."""
        from primus.backends.megatron.megatron_base_trainer import MegatronBaseTrainer
        
        trainer_cls = BackendRegistry.get_trainer_class("megatron_sft")
        self.assertTrue(
            issubclass(trainer_cls, MegatronBaseTrainer),
            "MegatronSFTTrainer should inherit from MegatronBaseTrainer"
        )

    @patch('primus.backends.megatron.megatron_base_trainer.log_rank_0')
    @patch('primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer._patch_parse_args')
    @patch('primus.backends.megatron.megatron_base_trainer.MegatronBaseTrainer._patch_megatron_runtime_hooks')
    def test_sft_trainer_initialization(self, mock_runtime_hooks, mock_parse_args, mock_log):
        """Test MegatronSFTTrainer can be initialized."""
        # Mock successful patching
        mock_parse_args.return_value = True
        
        # Create mock configs
        primus_config = MagicMock()
        module_config = SimpleNamespace(
            model="llama3_8B",
            framework="megatron"
        )
        backend_args = SimpleNamespace(
            sft_dataset_name="tatsu-lab/alpaca",
            sft_conversation_format="alpaca"
        )
        
        # Initialize trainer
        trainer = MegatronSFTTrainer(
            primus_config=primus_config,
            module_config=module_config,
            backend_args=backend_args
        )
        
        # Verify trainer attributes
        self.assertEqual(trainer.primus_config, primus_config)
        self.assertEqual(trainer.module_config, module_config)
        self.assertEqual(trainer.backend_args, backend_args)

    def test_adapter_selects_sft_trainer(self):
        """Test that MegatronAdapter selects SFT trainer based on module name."""
        from primus.backends.megatron.megatron_adapter import MegatronAdapter
        
        adapter = MegatronAdapter("megatron")
        
        # Test with sft_trainer module
        module_config = SimpleNamespace(name="sft_trainer")
        trainer_cls = adapter.load_trainer_class(module_config=module_config)
        self.assertEqual(
            trainer_cls.__name__,
            "MegatronSFTTrainer",
            "Should select MegatronSFTTrainer for sft_trainer module"
        )
        
        # Test with pre_trainer module (default)
        module_config = SimpleNamespace(name="pre_trainer")
        trainer_cls = adapter.load_trainer_class(module_config=module_config)
        self.assertEqual(
            trainer_cls.__name__,
            "MegatronPretrainTrainer",
            "Should select MegatronPretrainTrainer for pre_trainer module"
        )
        
        # Test with no module config (default)
        trainer_cls = adapter.load_trainer_class(module_config=None)
        self.assertEqual(
            trainer_cls.__name__,
            "MegatronPretrainTrainer",
            "Should default to MegatronPretrainTrainer"
        )


if __name__ == "__main__":
    unittest.main()
