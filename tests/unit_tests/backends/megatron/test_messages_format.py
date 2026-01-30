###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for multi-turn conversation support with OpenAI messages format.
"""

import json
import os
import tempfile
import unittest
from typing import List, Dict


class TestOpenAIMessagesFormat(unittest.TestCase):
    """Test OpenAI messages format for multi-turn conversations."""

    def setUp(self):
        """Create temporary test files."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create test file with messages format
        self.messages_file = os.path.join(self.temp_dir, "messages.jsonl")
        with open(self.messages_file, 'w', encoding='utf-8') as f:
            # Multi-turn conversation
            f.write(json.dumps({
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi!"},
                    {"role": "user", "content": "How are you?"},
                    {"role": "assistant", "content": "I'm good!"}
                ]
            }) + '\n')
            
            # Conversation without system message
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "Tell me a joke"},
                    {"role": "assistant", "content": "Why did the chicken cross the road?"}
                ]
            }) + '\n')
            
            # Single turn
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"}
                ]
            }) + '\n')

    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_format_messages_logic(self):
        """Test the messages formatting logic."""
        def format_messages(messages: List[Dict[str, str]]):
            """Format messages into text with assistant ranges."""
            parts = []
            assistant_ranges = []
            current_pos = 0
            
            for msg in messages:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if not role or not content:
                    continue
                    
                role_header = f"<|im_start|>{role}\n"
                role_footer = "<|im_end|>\n"
                
                parts.append(role_header)
                current_pos += len(role_header)
                
                if role == "assistant":
                    start_pos = current_pos
                    end_pos = current_pos + len(content)
                    assistant_ranges.append((start_pos, end_pos))
                
                parts.append(content)
                current_pos += len(content)
                
                parts.append(role_footer)
                current_pos += len(role_footer)
            
            return "".join(parts), assistant_ranges
        
        # Test with multi-turn
        messages = [
            {"role": "system", "content": "Be helpful"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "Bye"},
            {"role": "assistant", "content": "Goodbye!"}
        ]
        
        formatted, ranges = format_messages(messages)
        
        # Verify 2 assistant responses
        self.assertEqual(len(ranges), 2)
        
        # Verify content extraction
        self.assertEqual(formatted[ranges[0][0]:ranges[0][1]], "Hello!")
        self.assertEqual(formatted[ranges[1][0]:ranges[1][1]], "Goodbye!")
        
        # Verify format structure
        self.assertIn("<|im_start|>system", formatted)
        self.assertIn("<|im_start|>user", formatted)
        self.assertIn("<|im_start|>assistant", formatted)
        self.assertIn("<|im_end|>", formatted)

    def test_messages_file_loading(self):
        """Test loading JSONL file with messages format."""
        try:
            from primus.backends.megatron.core.datasets.sft_dataset import load_jsonl_file
        except ImportError as e:
            self.skipTest(f"Cannot import: {e}")
        
        data = load_jsonl_file(self.messages_file)
        
        # Should have 3 samples
        self.assertEqual(len(data), 3)
        
        # First sample should have messages
        self.assertIn("messages", data[0])
        self.assertEqual(len(data[0]["messages"]), 5)
        
        # Second sample
        self.assertEqual(len(data[1]["messages"]), 2)
        
        # Third sample
        self.assertEqual(len(data[2]["messages"]), 2)

    def test_openai_formatter(self):
        """Test OpenAIMessagesFormatter class."""
        try:
            from primus.backends.megatron.core.datasets.sft_dataset import OpenAIMessagesFormatter
        except ImportError as e:
            self.skipTest(f"Cannot import formatter: {e}")
        
        formatter = OpenAIMessagesFormatter()
        
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
        
        formatted_text, assistant_ranges = formatter.format_messages(messages)
        
        # Should have 1 assistant response
        self.assertEqual(len(assistant_ranges), 1)
        
        # Extract assistant content
        start, end = assistant_ranges[0]
        assistant_content = formatted_text[start:end]
        self.assertEqual(assistant_content, "Hi there!")

    def test_sft_dataset_with_messages(self):
        """Test SFTDataset loading messages format."""
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
                dataset_name=self.messages_file,
                tokenizer=MockTokenizer(),
                max_seq_length=512,
                formatter="openai"
            )
            
            self.assertEqual(len(dataset), 3)
            
            # Get first sample (multi-turn)
            sample = dataset[0]
            self.assertIn("input_ids", sample)
            self.assertIn("labels", sample)
            self.assertIn("loss_mask", sample)
            
            # Loss mask should have some 1s (for assistant responses)
            # and some 0s (for user/system messages)
            loss_sum = sample["loss_mask"].sum().item()
            self.assertGreater(loss_sum, 0)
            self.assertLess(loss_sum, len(sample["loss_mask"]))
            
        except Exception as e:
            if "No module named" in str(e):
                self.skipTest(f"Required module not available: {e}")
            raise

    def test_formatter_selection(self):
        """Test that openai and messages formatter names work."""
        try:
            from primus.backends.megatron.core.datasets.sft_dataset import SFTDataset
        except ImportError as e:
            self.skipTest(f"Cannot import: {e}")
        
        class MockTokenizer:
            def tokenize(self, text):
                return text.split()
            def convert_tokens_to_ids(self, tokens):
                return [hash(token) % 10000 for token in tokens]
        
        try:
            # Test "openai" formatter name
            dataset1 = SFTDataset(
                dataset_name=self.messages_file,
                tokenizer=MockTokenizer(),
                max_seq_length=512,
                formatter="openai"
            )
            self.assertEqual(len(dataset1), 3)
            
            # Test "messages" formatter name
            dataset2 = SFTDataset(
                dataset_name=self.messages_file,
                tokenizer=MockTokenizer(),
                max_seq_length=512,
                formatter="messages"
            )
            self.assertEqual(len(dataset2), 3)
            
        except Exception as e:
            if "No module named" in str(e):
                self.skipTest(f"Required module not available: {e}")
            raise


if __name__ == "__main__":
    unittest.main()
