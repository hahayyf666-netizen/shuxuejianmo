"""Evidence mapping with explicit proof gates. No uniform index-to-seconds fallback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Iterable

VALID_STATUSES = {"verified_text", "verified_time", "index_only", "unavailable"}


@dataclass(frozen=True)
class EvidenceRecord:
    sample_id: str
    modality: str
    official_seq_index: int
    mapping_status: str
    source_hash: str
    mapping_method: str
    mapping_version: str
    text_char_start: int | None = None
    text_char_end: int | None = None
    start_sec: float | None = None
    end_sec: float | None = None
    frame_pts_sec: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def raw_text_hash(raw_text: str) -> str:
    return hashlib.sha256(raw_text.encode("utf-8")).hexdigest()


def index_only(sample_id: str, modality: str, seq_index: int, source_hash: str, reason: str) -> EvidenceRecord:
    if modality not in ("text", "audio", "vision") or not 0 <= seq_index < 50:
        raise ValueError("invalid modality/index")
    return EvidenceRecord(sample_id, modality, seq_index, "index_only", source_hash,
                          "official_sequence_index", "q3-v1-preflight", note=reason)


def map_text_offsets(
    *, sample_id: str, raw_text: str, official_token_ids: list[int],
    replay_token_ids: list[int], replay_offsets: list[tuple[int, int]],
    content_indices: Iterable[int], tokenizer_revision: str,
    feature_row_provenance_verified: bool,
    provenance_document_sha256: str | None = None,
    official_attention: list[int] | None = None,
    replay_attention: list[int] | None = None,
) -> list[EvidenceRecord]:
    """Map only exact replayed IDs; row provenance is a separate required proof."""
    source_hash = raw_text_hash(raw_text)
    indices = list(content_indices)
    if not tokenizer_revision or len(official_token_ids) != 50 or len(replay_token_ids) != 50 or len(replay_offsets) != 50:
        return [index_only(sample_id, "text", i, source_hash, "tokenizer revision/length not verified") for i in indices]
    if official_token_ids != replay_token_ids:
        return [index_only(sample_id, "text", i, source_hash, "full 50-position token replay mismatch") for i in indices]
    if (official_attention is None or replay_attention is None or
            len(official_attention) != 50 or official_attention != replay_attention or
            set(official_attention) - {0, 1}):
        return [index_only(sample_id, "text", i, source_hash, "full attention-mask replay mismatch") for i in indices]
    active = [j for j, flag in enumerate(official_attention) if flag == 1]
    if (not active or active != list(range(len(active))) or
            official_token_ids[0] != 101 or official_token_ids[active[-1]] != 102 or
            indices != active[1:-1]):
        return [index_only(sample_id, "text", i, source_hash, "special-token/content-index structure mismatch") for i in indices]
    records = []
    for i in indices:
        if not 0 <= i < 50:
            raise ValueError("invalid official index")
        start, end = replay_offsets[i]
        exact = official_token_ids[i] == replay_token_ids[i] and 0 <= start < end <= len(raw_text)
        if not exact:
            records.append(index_only(sample_id, "text", i, source_hash, "token ID or character offset mismatch"))
        elif not feature_row_provenance_verified or not provenance_document_sha256 or len(provenance_document_sha256) != 64:
            records.append(EvidenceRecord(sample_id, "text", i, "index_only", source_hash,
                                          "tokenizer_replay_candidate", tokenizer_revision,
                                          text_char_start=start, text_char_end=end,
                                          note="token replay matches; official text feature row provenance document unverified"))
        else:
            records.append(EvidenceRecord(sample_id, "text", i, "verified_text", source_hash,
                                          "tokenizer_replay_plus_row_provenance", tokenizer_revision,
                                          text_char_start=start, text_char_end=end,
                                          note="provenance_document_sha256=" + provenance_document_sha256))
    return records


def map_av_support(
    *, sample_id: str, modality: str, seq_index: int, media_sha256: str,
    start_sec: float | None, end_sec: float | None, frame_pts_sec: float | None,
    mapping_method: str, mapping_version: str,
    shared_timeline_verified: bool, official_row_support_verified: bool,
    content_review_verified: bool, proof_manifest_sha256: str | None = None,
) -> EvidenceRecord:
    """Only emit verified_time when support of the official row is demonstrated."""
    if modality not in ("audio", "vision") or not 0 <= seq_index < 50:
        raise ValueError("invalid modality/index")
    if not media_sha256 or len(media_sha256) != 64:
        raise ValueError("media SHA-256 required")
    if not (shared_timeline_verified and official_row_support_verified and content_review_verified):
        return index_only(sample_id, modality, seq_index, media_sha256,
                          "timeline, official-row support, or content review unverified")
    if not proof_manifest_sha256 or len(proof_manifest_sha256) != 64:
        raise ValueError("verified AV support requires auditable proof manifest SHA-256")
    if not mapping_method or not mapping_version or start_sec is None or end_sec is None:
        raise ValueError("verified AV support requires method, version and time interval")
    if not 0 <= start_sec < end_sec or (frame_pts_sec is not None and not start_sec <= frame_pts_sec <= end_sec):
        raise ValueError("invalid presentation time support")
    if modality == "vision" and frame_pts_sec is None:
        raise ValueError("verified vision evidence requires frame PTS")
    return EvidenceRecord(sample_id, modality, seq_index, "verified_time", media_sha256,
                          mapping_method, mapping_version, start_sec=start_sec, end_sec=end_sec,
                          frame_pts_sec=frame_pts_sec, note="proof_manifest_sha256=" + proof_manifest_sha256)
