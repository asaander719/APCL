import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from apcl_repro.features import load_tensor
from apcl_repro.history import TrainingHistories, query_histories
from apcl_repro.models.APCL import APCL
from apcl_repro.runtime import evaluate_pairs, read_splits


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        self.rows = np.array([[0, 6, 1, 2], [0, 6, 2, 3], [1, 6, 1, 3],
                              [1, 7, 3, 2], [2, 7, 2, 1], [2, 6, 3, 1]])

    def test_query_history_ignores_positive_and_negative_labels(self):
        other = self.rows.copy(); other[:, 2:] = other[::-1, 2:]
        a = query_histories(self.rows, self.rows, 2, 2, 2026)
        b = query_histories(self.rows, other, 2, 2, 2026)
        np.testing.assert_array_equal(a, b)

    def test_training_sampling_is_repeatable_without_mutation(self):
        m = dict(u_pb_num=2, top_u=2, with_self_his=False, repeated_interact=False)
        sampler = TrainingHistories(self.rows, m)
        before = copy.deepcopy(dict(sampler.by_target))
        a = sampler.sample(self.rows, 42, 1)
        np.testing.assert_array_equal(a, sampler.sample(self.rows, 42, 1))
        self.assertEqual(before, dict(sampler.by_target))
        self.assertTrue((a != self.rows[:, 2, None]).all())

    def test_cross_split_filter_keeps_within_split_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = {s: Path(directory)/f'{s}.csv' for s in ('train','valid','test')}
            paths['train'].write_text('0,6,1,2\n0,6,1,2\n')
            paths['valid'].write_text('0,6,1,3\n1,6,2,3\n')
            paths['test'].write_text('1,6,2,1\n2,6,3,1\n')
            rows, source, _ = read_splits(paths)
            self.assertEqual(len(rows['train']), 2)
            self.assertEqual(source['valid'].tolist(), [2])
            self.assertEqual(source['test'].tolist(), [2])

    def test_corrupt_feature_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/'bad.pt'; p.write_bytes(b'PK\x03\x04truncated')
            with self.assertRaisesRegex(ValueError, 'Incomplete or corrupt'):
                load_tensor(p)

    def test_pair_auc_and_history_independence(self):
        torch.manual_seed(42)
        args = SimpleNamespace(weight_P=.9, hidden_dim=8, user_num=3, item_num=8,
                               with_visual=True, with_text=True, with_Nor=True, cos=True,
                               att=True, use_weighted_loss=False, temperature=1., CL=True,
                               visual_feature_dim=4, text_feature_dim=6, device='cpu',
                               dataset='Polyvore_519', b_PC=True, uu_w=10., uu_v_w=.5,
                               include_indirect_score=False)
        model = APCL(args, None, torch.rand(8,4), torch.rand(8,6))
        history = np.array([[1,2]] * len(self.rows))
        auc, delta, per = evaluate_pairs(model, self.rows, history, 'cpu', 4)
        _, other, _ = evaluate_pairs(model, self.rows, history[:, ::-1].copy(), 'cpu', 4)
        np.testing.assert_array_equal(delta, other)
        self.assertEqual(auc, per.mean())
        self.assertTrue(model.CL)
        # Identical positive/negative scores receive half credit.
        tied = self.rows.copy(); tied[:,3] = tied[:,2]
        tied_auc, _, _ = evaluate_pairs(model, tied, history, 'cpu', 4)
        self.assertEqual(tied_auc, .5)


if __name__ == '__main__':
    unittest.main()
