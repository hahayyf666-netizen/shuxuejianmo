import unittest

import numpy as np
import torch

from q3v1.data import AlignedSample, TrainScaler, validate_content_indices, load_split
from q3v1.explain import conditional_ig, exact_shapley, local_attribution_status, summarize_contributions
from q3v1.mapping import map_av_support, map_text_offsets, raw_text_hash
from q3v1.model import Q3Model
from q3v1.train_eval import TrainConfig, choose_architecture, fit_candidate, metrics


def tensors():
    torch.manual_seed(2029)
    xs = tuple(torch.randn(2, 50, dim, dtype=torch.float32) for dim in (768, 74, 35))
    mask = torch.zeros(2, 50, dtype=torch.bool)
    mask[0, 1:8] = True
    mask[1, 2:12] = True
    return xs, mask


class InteractionModel(torch.nn.Module):
    def forward(self, text_feat, audio_feat, vision_feat, validity_mask):
        t = text_feat[:, 0, 0]
        a = audio_feat[:, 0, 0]
        value = t + a - t * a
        return torch.stack((torch.zeros_like(value), value, -value), dim=1), value


class Q3PreflightTests(unittest.TestCase):
    def test_model_forward_backward_and_padding_invariance(self):
        xs, mask = tensors()
        for variant in ("B0", "B1"):
            model = Q3Model(variant).eval()
            self.assertLess(model.parameter_count, 1_000_000)
            logits, regression = model(*xs, mask)
            self.assertEqual(tuple(logits.shape), (2, 3))
            self.assertEqual(tuple(regression.shape), (2,))
            self.assertTrue(bool(((regression >= -3) & (regression <= 3)).all()))
            changed = [x.clone() for x in xs]
            for x in changed:
                x[~mask] = 1e3
            new_logits, new_regression = model(*changed, mask)
            torch.testing.assert_close(logits, new_logits, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(regression, new_regression, atol=1e-6, rtol=1e-6)
            loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])) + torch.abs(regression).mean()
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))

    def test_prediction_metadata_is_not_forward_input(self):
        xs, mask = tensors()
        model = Q3Model("B0").eval()
        first = model(*xs, mask)
        metadata_a = {"raw_text": "one", "id": "old", "token_ids": [101, 102]}
        metadata_b = {"raw_text": "different", "id": "new", "token_ids": [100, 102]}
        self.assertNotEqual(metadata_a, metadata_b)
        second = model(*xs, mask)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        with self.assertRaises(TypeError):
            model(*xs, mask, metadata_b)

    def test_content_mask_and_test_split_guard(self):
        bert = np.zeros((3, 50), dtype=np.int64)
        bert[0, :4] = [101, 2204, 2154, 102]
        bert[1, :4] = 1
        np.testing.assert_array_equal(validate_content_indices(bert), [1, 2])
        bert[0, 3] = 0
        with self.assertRaises(ValueError):
            validate_content_indices(bert)
        with self.assertRaises(PermissionError):
            load_split("unused.pkl", "test")

    def test_scaler_is_train_only_and_keeps_natural_zero(self):
        indices = np.array([1, 2], dtype=np.int64)
        def sample(value):
            return AlignedSample("x", "text", np.full((50, 768), value, np.float32),
                                 np.zeros((50, 74), np.float32), np.zeros((50, 35), np.float32),
                                 indices, 1, 0.0)
        scaler = TrainScaler().fit([sample(1), sample(3)], split="train")
        transformed = scaler.transform(sample(2))
        self.assertTrue(np.all(transformed[0][~transformed[3]] == 0))
        self.assertTrue(np.all(transformed[1] == 0))
        with self.assertRaises(PermissionError):
            TrainScaler().fit([sample(2)], split="valid")

    def test_shapley_additivity_and_unresolved_interaction(self):
        model = InteractionModel().eval()
        xs = [torch.zeros(1, 50, d) for d in (768, 74, 35)]
        xs[0][0, 0, 0] = 1
        xs[1][0, 0, 0] = 1
        xs = tuple(xs)
        ref = tuple(torch.zeros_like(x) for x in xs)
        mask = torch.zeros(1, 50, dtype=torch.bool)
        mask[0, 0] = True
        shapley = exact_shapley(model, xs, ref, mask, target="regression")
        self.assertAlmostEqual(shapley["phi"]["text"], 0.5)
        self.assertAlmostEqual(shapley["phi"]["audio"], 0.5)
        self.assertAlmostEqual(shapley["additivity_residual"], 0)
        ig = conditional_ig(model, xs, ref, mask, modality="text", target="regression", steps_schedule=(8,))
        self.assertEqual(ig["numerical_status"], "pass")
        self.assertEqual(local_attribution_status(shapley["phi"]["text"], ig, mask[0].numpy()),
                         "local_attribution_unresolved_interaction")
        classification = exact_shapley(model, xs, ref, mask, target="classification")
        self.assertEqual(classification["target_class"], 1)
        self.assertAlmostEqual(classification["additivity_residual"], 0, places=6)

    def test_primary_semantics(self):
        result = summarize_contributions({"text": -0.4, "audio": -0.2, "vision": -0.1}, target="classification")
        self.assertEqual(result["primary_influential_modality"], "text")
        self.assertEqual(result["primary_supporting_modality"], "NONE")
        flat = summarize_contributions({"text": 0, "audio": 0, "vision": 0}, target="classification")
        self.assertEqual(flat["primary_influential_modality"], "unresolved")
        reg = summarize_contributions({"text": -1, "audio": 0, "vision": 0.5}, target="regression")
        self.assertEqual(reg["direction_labels"]["text"], "push_negative")

    def test_mapping_fail_closed(self):
        raw = "good day"
        ids = [101, 2204, 2154, 102] + [0] * 46
        offsets = [(0, 0), (0, 4), (5, 8), (0, 0)] + [(0, 0)] * 46
        replay = map_text_offsets(sample_id="s", raw_text=raw, official_token_ids=ids,
                                  replay_token_ids=ids, replay_offsets=offsets,
                                  content_indices=[1, 2], tokenizer_revision="pinned-revision",
                                  feature_row_provenance_verified=False,
                                  official_attention=[1] * 4 + [0] * 46,
                                  replay_attention=[1] * 4 + [0] * 46)
        self.assertEqual([x.mapping_status for x in replay], ["index_only", "index_only"])
        confirmed = map_text_offsets(sample_id="s", raw_text=raw, official_token_ids=ids,
                                     replay_token_ids=ids, replay_offsets=offsets,
                                     content_indices=[1, 2], tokenizer_revision="pinned-revision",
                                     feature_row_provenance_verified=True,
                                     provenance_document_sha256="a" * 64,
                                     official_attention=[1] * 4 + [0] * 46,
                                     replay_attention=[1] * 4 + [0] * 46)
        self.assertEqual(confirmed[0].mapping_status, "verified_text")
        media_hash = raw_text_hash(raw)
        unknown = map_av_support(sample_id="s", modality="vision", seq_index=1, media_sha256=media_hash,
                                 start_sec=1.0, end_sec=2.0, frame_pts_sec=1.5,
                                 mapping_method="PTS", mapping_version="1", shared_timeline_verified=True,
                                 official_row_support_verified=False, content_review_verified=True)
        self.assertEqual(unknown.mapping_status, "index_only")
        with self.assertRaises(ValueError):
            map_av_support(sample_id="s", modality="vision", seq_index=1, media_sha256=media_hash,
                           start_sec=1.0, end_sec=2.0, frame_pts_sec=None,
                           mapping_method="PTS", mapping_version="1", shared_timeline_verified=True,
                           official_row_support_verified=True, content_review_verified=True,
                           proof_manifest_sha256="b" * 64)

    def test_metric_and_formal_training_guard(self):
        result = metrics(np.array([0, 1, 2]), np.array([0, 1, 2]),
                         np.array([-1.0, 0.0, 1.0]), np.array([-1.0, 0.0, 1.0]))
        self.assertAlmostEqual(result["accuracy"], 1.0)
        self.assertAlmostEqual(result["macro_f1"], 1.0)
        self.assertAlmostEqual(result["selection_J"], 0.0)
        with self.assertRaises(PermissionError):
            fit_candidate(train=[], valid=[], scaler=TrainScaler(), variant="B0", seed=2029,
                          checkpoint=__import__("pathlib").Path("unused.pt"), device=torch.device("cpu"))
        b0 = [{"seed": seed, "best_J": 0.30000} for seed in (2029, 2030, 2031)]
        b1 = [{"seed": seed, "best_J": 0.29995} for seed in (2029, 2030, 2031)]
        self.assertEqual(choose_architecture({"B0": b0, "B1": b1}), "B0")


if __name__ == "__main__":
    unittest.main()
