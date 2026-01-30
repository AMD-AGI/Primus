###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for offline JSONL dataset support in SFT trainer.
"""

import json
import os
import tempfile
import unittest


class TestJSONLDatasetLoading(unittest.TestCase):
    """Test JSONL and JSON file loading for SFT datasets."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test JSONL file
        self.jsonl_file = os.path.join(self.temp_dir, "test.jsonl")
        with open(self.jsonl_file, 'w', encoding='utf-8') as f:
            f.write('{"instruction": "Test 1", "response": "Response 1"}\n')
            f.write('{"instruction": "Test 2", "response": "Response 2"}\n')
            f.write('{"instruction": "Test 3", "response": "Response 3"}\n')
        
        # Create test JSON file
        self.json_file = os.path.join(self.temp_dir, "test.json")
        with open(self.json_file, 'w', encoding='utf-8') as f:
            json.dump([
                {"instruction": "JSON Test 1", "response": "JSON Response 1"},
                {"instruction": "JSON Test 2", "response": "JSON Response 2"}
            ], f)

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_load_jsonl_file(self):
        """Test loading JSONL file."""
        from primus.backends.megatron.core.datasets.sft_dataset import load_jsonl_file
        
        data = load_jsonl_file(self.jsonl_file)
        
        self.assertEqual(len(data), 3)
        self.assertEqual(data[0]["instruction"], "Test 1")
        self.assertEqual(data[0]["response"], "Response 1")
        self.assertEqual(data[2]["instruction"], "Test 3")

    def test_load_jsonl_file_not_found(self):
        """Test error handling for missing file."""
        from primus.backends.megatron.core.datasets.sft_dataset import load_jsonl_file
        
        with self.assertRaises(FileNotFoundError):
            load_jsonl_file("/nonexistent/file.jsonl")

    def test_load_jsonl_file_invalid_json(self):
        """Test error handling for invalid JSON."""
        from primus.backends.megatron.core.datasets.sft_dataset import load_jsonl_file
        
        invalid_file = os.path.join(self.temp_dir, "invalid.jsonl")
        with open(invalid_file, 'w') as f:
            f.write('{"valid": "json"}\n')
            f.write('invalid json line\n')
            f.write('{"another": "valid"}\n')
        
        with self.assertRaises(json.JSONDecodeError):
            load_jsonl_file(invalid_file)

    def test_sft_dataset_with_jsonl(self):
        """Test SFTDataset can load from JSONL file."""
        try:
            from primus.backends.megatron.core.datasets.sft_dataset import SFTDataset
        except ImportError as e:
            self.skipTest(f"Cannot import SFTDataset: {e}")
        
        # Create mock tokenizer
        class MockTokenizer:
            def tokenize(self, text):
                return text.split()
            
            def convert_tokens_to_ids(self, tokens):
                return [hash(token) % 10000 for token in tokens]
        
        try:
            dataset = SFTDataset(
                dataset_name=self.jsonl_file,
                tokenizer=MockTokenizer(),
                max_seq_length=512,
                formatter="alpaca"
            )
            
            self.assertEqual(len(dataset), 3)
            
            # Get a sample
            sample = dataset[0]
            self.assertIn("input_ids", sample)
            self.assertIn("labels", sample)
            self.assertIn("loss_mask", sample)
            
        except Exception as e:
            # If torch or datasets not available, skip
            if "No module named" in str(e):
                self.skipTest(f"Required module not available: {e}")
            raise

    def test_sft_dataset_with_json(self):
        """Test SFTDataset can load from JSON array file."""
        try:
            from primus.backends.megatron.core.datasets.sft_dataset import SFTDataset
        except ImportError as e:
            self.skipTest(f"Cannot import SFTDataset: {e}")
        
        # Create mock tokenizer
        class MockTokenizer:
            def tokenize(self, text):
                return text.split()
            
            def convert_tokens_to_ids(self, tokens):
                return [hash(token) % 10000 for token in tokens]
        
        try:
            dataset = SFTDataset(
                dataset_name=self.json_file,
                tokenizer=MockTokenizer(),
                max_seq_length=512,
                formatter="alpaca"
            )
            
            self.assertEqual(len(dataset), 2)
            
            # Verify data was loaded correctly
            self.assertEqual(
                dataset.dataset[0]["instruction"],
                "JSON Test 1"
            )
            
        except Exception as e:
            # If torch or datasets not available, skip
            if "No module named" in str(e):
                self.skipTest(f"Required module not available: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
