import unittest
from unittest.mock import patch

from primus.cli.main import main
from primus.cli.registry import CommandRegistry


class TestCLI(unittest.TestCase):
    def setUp(self):
        """Set up the test environment."""
        CommandRegistry.discover_commands()

    def test_command_discovery(self):
        """Test that all commands are discovered and registered."""
        commands = CommandRegistry.get_all_commands()
        self.assertIn("train", commands)
        self.assertIn("benchmark", commands)

    @patch("sys.argv", ["primus", "train", "pretrain", "--config", "exp.yaml"])
    @patch("primus.cli.subcommands.train.TrainCommand.run")
    def test_train_command(self, mock_run):
        """Test the train command execution."""
        mock_run.return_value = None
        main()
        mock_run.assert_called_once()

    @patch("sys.argv", ["primus", "benchmark", "gemm", "--config", "gemm.yaml"])
    @patch("primus.cli.subcommands.benchmark.BenchmarkCommand.run")
    def test_benchmark_command(self, mock_run):
        """Test the benchmark command execution."""
        mock_run.return_value = None
        main()
        mock_run.assert_called_once()

    @patch("sys.argv", ["primus", "unknown"])
    def test_unknown_command(self):
        """Test handling of unknown commands."""
        with self.assertRaises(SystemExit) as cm:
            main()
        self.assertEqual(cm.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
