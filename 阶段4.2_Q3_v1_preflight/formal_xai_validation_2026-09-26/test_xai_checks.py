import json
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE if (HERE / "q3v1").is_dir() else HERE.parent))
from q3v1.explain import conditional_ig, local_attribution_status
from q3v1.model import Q3Model
from xai_checks import (fixed_target_ig, stable_rng, rank_positions, grouped_bootstrap,
                        position_comparison, perturbation_rows, replace_and_evaluate)

CONFIG = json.loads((HERE / "validation_contract.json").read_text())
torch.set_num_threads(2)


class LinearProbe(torch.nn.Module):
    def forward(self, text_feat, audio_feat, vision_feat, validity_mask):
        value = (text_feat[:, :, 0] * validity_mask).sum(-1)
        return torch.stack((torch.zeros_like(value), value, -value), -1), value


def inputs():
    xs = tuple(torch.zeros(1, 50, d) for d in (768, 74, 35))
    xs[0][0, 1, 0], xs[0][0, 4, 0] = 1, 2
    mask = torch.zeros(1, 50, dtype=torch.bool)
    mask[0, [1, 4]] = True
    ref = tuple(torch.zeros_like(x) for x in xs)
    return xs, ref, mask


class XaiChecks(unittest.TestCase):
    def test_fixed_target_ig_allows_non_predicted_original_class(self):
        xs, ref, mask = inputs()
        model = LinearProbe()
        with self.assertRaises(ValueError):
            conditional_ig(model, xs, ref, mask, modality="text", target="classification", target_class=0)
        out = fixed_target_ig(model, xs, ref, mask, modality="text", target="classification", target_class=0, config=CONFIG)
        self.assertEqual(out["target_class"], 0)
        self.assertEqual(out["numerical_status"], "pass")
        self.assertLess(out["position_scores"].sum(), 0)
        self.assertLess(abs(out["completeness_residual"]), 1e-5)

    def test_adapter_equivalence_and_linear_exact_answer(self):
        xs, ref, mask = inputs()
        model = LinearProbe()
        a = conditional_ig(model, xs, ref, mask, modality="text", target="regression")
        b = fixed_target_ig(model, xs, ref, mask, modality="text", target="regression", target_class=1, config=CONFIG)
        np.testing.assert_allclose(a["position_scores"], b["position_scores"], atol=1e-6)
        np.testing.assert_allclose(b["position_scores"][[1, 4]], [1, 2], atol=1e-6)
        self.assertTrue(np.all(b["position_scores"][~mask[0].numpy()] == 0))

    def test_real_architecture_adapter_equivalence(self):
        xs, ref, mask = inputs()
        torch.manual_seed(2029)
        model = Q3Model("B0").eval()
        c = int(model(*xs, mask)[0].argmax(-1))
        a = conditional_ig(model, xs, ref, mask, modality="text", target="classification")
        b = fixed_target_ig(model, xs, ref, mask, modality="text", target="classification", target_class=c, config=CONFIG)
        np.testing.assert_allclose(a["position_scores"], b["position_scores"], atol=1e-7, rtol=1e-6)

    def test_ranking_masks_and_ties(self):
        scores = np.full(50, 1e9)
        scores[[1, 4, 8]] = [-2, 2, 1]
        np.testing.assert_array_equal(rank_positions(scores, [8, 4, 1]), [1, 4, 8])

    def test_perturbations_keep_structure_and_do_not_mutate_input(self):
        xs, ref, mask = inputs()
        original = xs[0].clone()
        result = replace_and_evaluate(LinearProbe(), xs, ref, mask, "text", [[4], [1]], "regression", 0)
        np.testing.assert_allclose(result, [1, 2])
        torch.testing.assert_close(xs[0], original, atol=0, rtol=0)
        self.assertEqual(int(mask.sum()), 2)

    def test_same_size_random_controls_and_no_false_positive_support(self):
        xs, ref, mask = inputs()
        scores = np.zeros(50); scores[[1, 4]] = [-1, -2]
        ig = {"position_scores": scores}
        rows = perturbation_rows(LinearProbe(), xs, ref, mask, ig, "localized_feature_position",
                                 "video$_$1", "classification", 1, "text", CONFIG)
        for row in rows:
            self.assertEqual(row["k"], 1)
            self.assertEqual(len(row["random_positions"]), 50)
            self.assertTrue(all(len(x) == 1 and set(x) <= {1, 4} for x in row["random_positions"]))
            self.assertEqual(row["positive_support_count"], 0)
            self.assertIsNone(row["positive_support_signed_drop"])

    def test_weak_and_unresolved_do_not_get_topk(self):
        xs, ref, mask = inputs()
        ig = {"position_scores": np.zeros(50), "numerical_status": "pass"}
        status = local_attribution_status(0.5, ig, mask[0].numpy())
        self.assertEqual(status, "local_attribution_unresolved_interaction")
        rows = perturbation_rows(LinearProbe(), xs, ref, mask, ig, status, "v$_$1", "regression", 0, "text", CONFIG)
        self.assertTrue(all(not r["eligible"] and "top_positions" not in r for r in rows))
        self.assertEqual(position_comparison(ig, ig, np.array([1, 4]), CONFIG)["status"], "weak_or_unlocalized")

    def test_random_reproducibility_and_cluster_bootstrap(self):
        np.testing.assert_array_equal(stable_rng(2029, "v$_$1").integers(100, size=20), stable_rng(2029, "v$_$1").integers(100, size=20))
        rows = [{"video_id": g, "eligible": True, "paired_difference": 0.5} for g in ("a", "a", "b")]
        result = grouped_bootstrap(rows, CONFIG, "key")
        self.assertEqual(result["n_video_groups"], 2)
        self.assertEqual(result["ci95"], [0.5, 0.5])
        self.assertEqual(result["evidence"], "above_random")
        result = grouped_bootstrap(rows[:2], CONFIG, "key")
        self.assertIsNone(result["ci95"])
        rows[0]["eligible"] = False
        self.assertEqual(grouped_bootstrap(rows, CONFIG, "key")["n_skipped"], 1)


if __name__ == "__main__":
    unittest.main()
