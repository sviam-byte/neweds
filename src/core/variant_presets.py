from __future__ import annotations

"""Пресеты наборов вариантов связности и их разворачивание."""

from typing import List, Tuple


# Алиасы коротких имён → каноническое имя в реестре (registry.py).
# Позволяет использовать сокращения в пресетах и пользовательском вводе.
VARIANT_ALIASES: dict[str, str] = {
    "corr_full": "correlation_full",
    "corr_partial": "correlation_partial",
    "corr_directed": "correlation_directed",
    "coh_full": "coherence_full",
    "coh_partial": "coherence_partial",
    "fftcoh_full": "coherence_full",  # legacy alias
}


def _resolve_alias(name: str) -> str:
    """Возвращает каноническое имя метрики (или само имя, если алиаса нет)."""
    return VARIANT_ALIASES.get(name, name)


PRESETS = {
    # максимально безопасный по времени и интерпретации
    "basic": [
        "corr_full",
        "coh_full",
        "mutinf_full",
    ],
    # частотные/сложностные
    "spectral": [
        "coh_full",
        "coh_partial",
    ],
    "entropy": [
        "mutinf_full",
        "mutinf_partial",
    ],
    # нелинейные (без параметров)
    "nonlinear": [
        "dcor_full",
        "dcor_partial",
        "ordinal_full",
    ],
    # направленные/каузальные
    "causal": [
        "granger_directed",
        "te_directed",
        "ah_directed",
        "dcor_directed",
        "ordinal_directed",
    ],
    # «всё адекватное»: без слишком экспериментальных комбинаций
    "full": [
        "corr_full",
        "corr_partial",
        "coh_full",
        "mutinf_full",
        "mutinf_partial",
        "dcor_full",
        "ordinal_full",
        "granger_directed",
        "te_directed",
        "ah_directed",
    ],
    "all": [
        "corr_full",
        "corr_partial",
        "coh_full",
        "coh_partial",
        "mutinf_full",
        "mutinf_partial",
        "dcor_full",
        "dcor_partial",
        "dcor_directed",
        "ordinal_full",
        "ordinal_directed",
        "granger_full",
        "granger_partial",
        "granger_directed",
        "te_full",
        "te_partial",
        "te_directed",
        "ah_full",
        "ah_partial",
        "ah_directed",
    ],
}


def expand_variants(tokens: List[str]) -> Tuple[List[str], str]:
    """Разворачивает смесь пресетов и явных variant-ов.

    Args:
        tokens: список токенов, где каждый элемент — либо имя пресета,
            либо конкретный variant.

    Returns:
        Кортеж `(variants, explain_text)`:
          - `variants`: уникальный список в исходном порядке.
          - `explain_text`: пояснение, какие пресеты развернулись во что.
    """
    out: List[str] = []
    explain: List[str] = []

    for token in tokens:
        key = str(token).strip().lower()
        if not key:
            continue
        if key in PRESETS:
            vs = PRESETS[key]
            explain.append(f"preset '{key}' -> {', '.join(vs)}")
            out.extend(vs)
        else:
            # Пробуем разрешить алиас, чтобы пользователь мог писать 'corr_full'
            out.append(_resolve_alias(key))

    # unique with stable order
    seen = set()
    uniq: List[str] = []
    for variant in out:
        if variant not in seen:
            uniq.append(variant)
            seen.add(variant)

    if not explain:
        explain.append("variants задан списком напрямую")

    return uniq, "\n".join(explain)
