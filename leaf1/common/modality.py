"""Helpers for manifest modality tokens."""


MODALITY_ALIASES = {
    "cfdna": "CFDNA",
    "cfdna-wgs": "CFDNA",
    "tcga": "TCGA",
    "tcga_atac": "TCGA",
    "bulk_atac": "TCGA",
    "scatac": "SCATAC",
    "single_cell_atac": "SCATAC",
}


def resolve_modality_token(tokenizer, modality):
    """Return the bracketed modality token known by the tokenizer.

    Manifests used for the paper store modalities as CFDNA, TCGA, and SCATAC.
    The public examples sometimes use lowercase names, so normalize common aliases
    and fail loudly if a manifest value is not represented in the tokenizer.
    """
    raw = str(modality).strip()
    canonical = MODALITY_ALIASES.get(raw, MODALITY_ALIASES.get(raw.lower(), raw))
    token = f"[{canonical}]"

    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id != tokenizer.unk_token_id:
        return token

    valid = ", ".join(f"[{v}]" for v in sorted(set(MODALITY_ALIASES.values())))
    raise ValueError(
        f"Unknown modality {modality!r}: token {token!r} is not in the tokenizer. "
        f"Use one of {valid}, or add the modality token to the tokenizer."
    )
