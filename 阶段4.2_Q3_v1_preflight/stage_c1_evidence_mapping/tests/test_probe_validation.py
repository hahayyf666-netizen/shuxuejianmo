"""Regression checks for the pre-cast text_bert validation gate."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from probe_tokenizer_mapping import validate_text_bert


class TextBertValidationTest(unittest.TestCase):
    @staticmethod
    def valid_array() -> np.ndarray:
        value = np.zeros((3, 50), dtype=np.float64)
        value[0, :3] = [101, 2023, 102]
        value[1, :3] = 1
        return value

    def test_valid_integer_valued_float_is_converted_after_validation(self) -> None:
        checked = validate_text_bert(self.valid_array(), sample_id="01", vocab_size=30522)
        self.assertEqual(checked.dtype, np.int64)
        self.assertEqual(checked[0, :3].tolist(), [101, 2023, 102])

    def test_rejects_values_that_integer_cast_would_hide(self) -> None:
        cases = {
            "wrong_shape": lambda a: a[:, :-1],
            "nan": lambda a: self.changed(a, (0, 1), np.nan),
            "infinity": lambda a: self.changed(a, (0, 1), np.inf),
            "fractional_token": lambda a: self.changed(a, (0, 1), 2023.5),
            "fractional_attention": lambda a: self.changed(a, (1, 1), 0.5),
            "fractional_segment": lambda a: self.changed(a, (2, 1), 0.5),
            "nonbinary_attention": lambda a: self.changed(a, (1, 1), 2),
            "nonprefix_attention": lambda a: self.changed(a, (1, 1), 0),
            "segment_one": lambda a: self.changed(a, (2, 1), 1),
            "wrong_cls": lambda a: self.changed(a, (0, 0), 2023),
            "wrong_sep": lambda a: self.changed(a, (0, 2), 2023),
            "unexpected_inner_sep": lambda a: self.changed(a, (0, 1), 102),
            "no_content_token": self.without_content,
            "nonzero_padding": lambda a: self.changed(a, (0, 3), 2023),
            "out_of_vocab": lambda a: self.changed(a, (0, 1), 30522),
            "object_dtype": lambda a: a.astype(object),
            "unsigned_overflow": lambda a: self.changed(a.astype(np.uint64), (0, 1), 2**63),
        }
        for label, mutate in cases.items():
            with self.subTest(label=label):
                with self.assertRaises(ValueError):
                    validate_text_bert(mutate(self.valid_array()), sample_id="01", vocab_size=30522)

    @staticmethod
    def changed(value: np.ndarray, index: tuple[int, int], replacement: object) -> np.ndarray:
        value[index] = replacement
        return value

    @staticmethod
    def without_content(value: np.ndarray) -> np.ndarray:
        value[0, 1:3] = [102, 0]
        value[1, 2] = 0
        return value


if __name__ == "__main__":
    unittest.main()
