import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from nimble.paths import PROJECT_ROOT
from nimble.training.augmented_data import validate_augmented_data
from nimble.training.schema_data import fingerprint, read_rows
from nimble.training.verify_dataset import verify


class CanonicalDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = PROJECT_ROOT / 'data'
        cls.rows = read_rows(cls.directory / 'train.jsonl')
        cls.manifest = json.loads((cls.directory / 'manifest.json').read_text())

    def test_release_is_self_contained_and_holdout_is_unchanged(self):
        result = verify(self.directory)
        self.assertEqual((result['training_rows'], result['holdout_rows']), (2826, 324))
        self.assertEqual(result['sha256']['eval.jsonl'],
                         '8e9e48b8de5206593912ae01ddc95bd77e40ad2ecf4c9292c1711290eca0d896')
        self.assertEqual({p.name for p in self.directory.iterdir()},
                         {'train.jsonl', 'eval.jsonl', 'manifest.json'})

    def test_changed_holdout_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('train.jsonl', 'eval.jsonl', 'manifest.json'):
                shutil.copyfile(self.directory / name, root / name)
            with (root / 'eval.jsonl').open('a') as file:
                file.write('\n')
            with self.assertRaisesRegex(ValueError, 'Frozen dataset bytes changed: eval.jsonl'):
                verify(root)

    def test_review_reconstruction_rejects_tampering_even_with_new_training_hash(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['reference']['reason'] = 'Changed after review'
        manifest = {**self.manifest, 'train_sha256': fingerprint(rows)}
        with self.assertRaisesRegex(ValueError, 'Reviewed release bytes changed'):
            validate_augmented_data(rows, manifest)


if __name__ == '__main__':
    unittest.main()
