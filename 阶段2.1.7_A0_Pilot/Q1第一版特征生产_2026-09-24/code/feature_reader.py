from __future__ import annotations

from pathlib import Path

import numpy as np


REQUIRED_FIELDS = {
    "schema_version", "sample_key", "official_text", "alignment_mode", "alignment_granularity",
    "text_audio_correspondence", "text_av_time_mapping_status", "text_word_feat", "text_word_valid",
    "raw_audio_lld_values", "raw_audio_lld_start_sec", "raw_audio_lld_end_sec", "raw_audio_lld_center_sec",
    "raw_audio_lld_valid", "raw_video_blendshape_values", "raw_video_pts_sec", "raw_video_support_start_sec",
    "raw_video_support_end_sec", "raw_video_face_feature_valid", "text_present", "audio_present", "video_present",
    "audio_observation_length", "video_observation_length", "text_sequence_length",
}


def load_q1_feature(path: str | Path) -> dict[str, np.ndarray]:
    """Load one q1-feature-v1.0 NPZ without pickle and check its public schema basics."""
    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as data:
        missing = REQUIRED_FIELDS.difference(data.files)
        if missing:
            raise ValueError(f"{archive_path}: missing required fields: {sorted(missing)}")
        arrays = {name: data[name].copy() for name in data.files}
    schema = str(arrays["schema_version"].item())
    if schema != "q1-feature-v1.0":
        raise ValueError(f"{archive_path}: unsupported schema_version={schema!r}")
    key = str(arrays["sample_key"].item())
    if not key:
        raise ValueError(f"{archive_path}: sample_key is empty")
    for name in ("text_word_feat", "raw_audio_lld_values", "raw_video_blendshape_values"):
        value = arrays[name]
        if value.dtype.hasobject or not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{archive_path}: {name} is not a numeric array")
        if not np.isfinite(value).all():
            raise ValueError(f"{archive_path}: {name} contains non-finite values")
    if arrays["text_word_feat"].shape[0] != int(arrays["text_sequence_length"].item()):
        raise ValueError(f"{archive_path}: text feature length differs from text_sequence_length")
    if arrays["raw_audio_lld_values"].shape[0] != int(arrays["audio_observation_length"].item()):
        raise ValueError(f"{archive_path}: audio feature length differs from audio_observation_length")
    if arrays["raw_video_blendshape_values"].shape[0] != int(arrays["video_observation_length"].item()):
        raise ValueError(f"{archive_path}: video feature length differs from video_observation_length")
    if arrays["raw_audio_lld_values"].shape[1] != 25 or arrays["raw_video_blendshape_values"].shape[1] != 52:
        raise ValueError(f"{archive_path}: unexpected native audio or video feature dimension")
    if arrays["text_word_feat"].shape[1] != 768:
        raise ValueError(f"{archive_path}: unexpected text feature dimension")
    return arrays


def summarize_q1_feature(arrays: dict[str, np.ndarray]) -> dict[str, object]:
    """Return a JSON-friendly summary while keeping each native sequence length independent."""
    return {
        "schema_version": str(arrays["schema_version"].item()),
        "sample_key": str(arrays["sample_key"].item()),
        "official_text": str(arrays["official_text"].item()),
        "alignment_mode": str(arrays["alignment_mode"].item()),
        "alignment_granularity": str(arrays["alignment_granularity"].item()),
        "text_audio_correspondence": str(arrays["text_audio_correspondence"].item()),
        "text_av_time_mapping_status": str(arrays["text_av_time_mapping_status"].item()),
        "text_present": int(arrays["text_present"].item()),
        "audio_present": int(arrays["audio_present"].item()),
        "video_present": int(arrays["video_present"].item()),
        "audio_speech_valid": int(arrays["audio_speech_valid"].item()),
        "face_feature_valid": int(arrays["face_feature_valid"].item()),
        "text_sequence_shape": list(arrays["text_word_feat"].shape),
        "text_valid_words": int(arrays["text_word_valid"].sum()),
        "audio_native_shape": list(arrays["raw_audio_lld_values"].shape),
        "audio_valid_windows": int(arrays["raw_audio_lld_valid"].sum()),
        "video_native_shape": list(arrays["raw_video_blendshape_values"].shape),
        "video_frames_with_face_features": int(arrays["raw_video_face_feature_valid"].sum()),
        "word_audio_shape": list(arrays["word_audio_feat"].shape),
        "word_audio_valid": int(arrays["word_audio_valid"].sum()),
        "word_vision_shape": list(arrays["word_vision_feat"].shape),
        "word_vision_valid": int(arrays["word_vision_valid"].sum()),
        "audio_presentation_coverage_sec": arrays["audio_presentation_coverage_sec"].astype(float).tolist(),
        "video_presentation_coverage_sec": arrays["video_presentation_coverage_sec"].astype(float).tolist(),
        "shared_t0_sec": float(arrays["shared_t0_sec"].item()),
    }
