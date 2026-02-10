
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

class TestTransferLoading(unittest.TestCase):
    def setUp(self):
        # We need to mock imports inside bioplausible.hyperopt.experiment
        # Since we can't easily unimport modules, we patch where the class is defined

        # Patch dependencies for TrialRunner instantiation
        self.patches = [
            patch("bioplausible.hyperopt.experiment.GLOBAL_CONFIG", MagicMock(epochs=1)),
            patch("bioplausible.hyperopt.experiment.create_task", return_value=MagicMock(input_dim=10, output_dim=2)),
            patch("bioplausible.hyperopt.experiment.ExperimentTracker"), # Prevent tracker init
            patch("bioplausible.hyperopt.experiment.ExperimentArchiver"),
        ]

        for p in self.patches:
            p.start()

        from bioplausible.hyperopt.experiment import TrialRunner
        self.runner = TrialRunner(storage=MagicMock(), task="mnist", quick_mode=True)

    def tearDown(self):
        for p in self.patches:
            p.stop()

    @patch("bioplausible.hyperopt.experiment.load_weights")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.iterdir")
    def test_load_transfer_weights_directory(self, mock_iterdir, mock_exists, mock_load_weights):
        # Setup: Artifact exists as a directory
        mock_exists.return_value = True

        artifact_path = MagicMock(spec=Path)
        artifact_path.name = "trial_123_model"
        artifact_path.is_dir.return_value = True

        # Mock the existence of model.pt inside the directory
        model_pt_path = MagicMock(spec=Path)
        model_pt_path.exists.return_value = True
        artifact_path.__truediv__.return_value = model_pt_path

        mock_iterdir.return_value = [artifact_path]

        model = MagicMock()
        config = {"freeze_layers": True}

        self.runner._load_transfer_weights(123, model, config)

        # Verify load_weights was called
        mock_load_weights.assert_called_once()
        # Check arguments (args[0] is model, args[1] is path)
        self.assertEqual(mock_load_weights.call_args[0][0], model)
        self.assertEqual(mock_load_weights.call_args[1]['freeze_layers'], True)

    @patch("bioplausible.hyperopt.experiment.load_weights")
    @patch("zipfile.ZipFile")
    @patch("pathlib.Path.exists")
    @patch("pathlib.Path.iterdir")
    @patch("tempfile.TemporaryDirectory")
    def test_load_transfer_weights_zip(self, mock_temp_dir, mock_iterdir, mock_exists, mock_zipfile, mock_load_weights):
        # Setup: Artifact exists as a zip
        mock_exists.return_value = True

        # Setup temp dir context manager
        temp_dir_path = "/tmp/test_ctx"
        mock_temp_dir.return_value.__enter__.return_value = temp_dir_path

        artifact_path = MagicMock(spec=Path)
        artifact_path.name = "trial_456_model.zip"
        artifact_path.suffix = ".zip"
        artifact_path.is_dir.return_value = False

        mock_iterdir.return_value = [artifact_path]

        # Mock zipfile context manager
        mock_zip = MagicMock()
        mock_zipfile.return_value.__enter__.return_value = mock_zip

        model = MagicMock()
        config = {}

        # We need to mock the path concatenation inside the method
        # The method does: found_path = temp_path / "model.pt"
        # Since temp_path is a Path object created from the string, we need to ensure it behaves correctly or mock Path
        # The real Path object works fine with strings, but verifying the check 'if found_path.exists():' requires patching

        # Just ensure load_weights is called. We can mock Path inside the function via sys.modules or just assume Path works
        # The easiest way to verify the 'exists' check passes is to mock load_weights call.

        # Let's mock Path inside the module to control .exists()
        with patch("bioplausible.hyperopt.experiment.Path") as mock_path_cls:
             # Make sure the initial artifacts dir check passes
            mock_path_instance = MagicMock()
            mock_path_instance.exists.return_value = True
            mock_path_cls.return_value = mock_path_instance

            # When creating temp_path from temp_dir_path
            mock_temp_path = MagicMock()

            # When doing temp_path / "model.pt"
            mock_model_path = MagicMock()
            mock_model_path.exists.return_value = True
            mock_temp_path.__truediv__.return_value = mock_model_path

            # Return specific mocks based on input
            def side_effect(arg):
                if arg == "artifacts":
                    return mock_path_instance
                if arg == temp_dir_path:
                    return mock_temp_path
                return MagicMock()

            mock_path_cls.side_effect = side_effect

            self.runner._load_transfer_weights(456, model, config)

        # Verify zip extraction
        mock_zip.extract.assert_called()

        # Verify load_weights called
        mock_load_weights.assert_called_once()

    @patch("pathlib.Path.exists")
    def test_load_transfer_weights_no_artifacts_dir(self, mock_exists):
        # Artifacts dir does not exist
        mock_exists.return_value = False

        model = MagicMock()
        config = {}

        # Should just print warning and return, not raise
        self.runner._load_transfer_weights(999, model, config)

if __name__ == "__main__":
    unittest.main()
