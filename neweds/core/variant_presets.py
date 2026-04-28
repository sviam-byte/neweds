from __future__ import annotations

"""Пресеты наборов вариантов связности и их разворачивание."""


from ..metrics.registry import METRICS_REGISTRY

# Алиасы коротких имён → каноническое имя в реестре (registry.py).
# Позволяет использовать сокращения в пресетах и пользовательском вводе.
VARIANT_ALIASES: dict[str, str] = {
    "corr_full": "correlation_full",
    "corr_partial": "correlation_partial",
    "corr_directed": "correlation_directed",
    "coh_full": "coherence_full",
    "coh_partial": "coherence_partial",
    "fftcoh_full": "coherence_full",  # short alias
}


def _resolve_alias(name: str) -> str:
    """Возвращает каноническое имя метрики (или само имя, если алиаса нет)."""
    return VARIANT_ALIASES.get(name, name)


OPT_IN_EXPERIMENTAL_VARIANTS = {"ah_full", "ah_partial", "ah_directed"}
ALL_REGISTRY_VARIANTS = list(METRICS_REGISTRY.keys())
PUBLIC_REGISTRY_VARIANTS = [
    v for v in ALL_REGISTRY_VARIANTS if v not in OPT_IN_EXPERIMENTAL_VARIANTS
]


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
        "ordinal_full",
    ],
    # нелинейные
    "nonlinear": [
        "dcor_full",
        "dcor_partial",
        "ordinal_full",
        "h2_full",
        "h2_partial",
    ],
    # направленные/каузальные — только реально существующие variant-ы
    "causal": [
        variant
        for variant in [
            "correlation_directed",
            "h2_directed",
            "granger_full",
            "granger_partial",
            "te_full",
            "te_partial",
            "dcor_directed",
            "ordinal_directed",
        ]
        if variant in ALL_REGISTRY_VARIANTS
    ],
    # «полный разумный»
    "full": [
        variant
        for variant in [
            "correlation_full",
            "correlation_spearman",
            "correlation_kendall",
            "correlation_partial",
            "correlation_directed",
            "coherence_full",
            "coherence_partial",
            "mutinf_full",
            "mutinf_partial",
            "dcor_full",
            "dcor_partial",
            "ordinal_full",
            "ordinal_directed",
            "granger_full",
            "granger_partial",
            "te_full",
            "te_partial",
        ]
        if variant in ALL_REGISTRY_VARIANTS
    ],
    # абсолютно всё из живого реестра
    "all": list(PUBLIC_REGISTRY_VARIANTS),
}


def expand_variants(tokens: list[str]) -> tuple[list[str], str]:
    """Разворачивает смесь пресетов и явных variant-ов.

    Args:
        tokens: список токенов, где каждый элемент — либо имя пресета,
            либо конкретный variant.

    Returns:
        Кортеж `(variants, explain_text)`:
          - `variants`: уникальный список в исходном порядке.
          - `explain_text`: пояснение, какие пресеты развернулись во что.
    """
    out: list[str] = []
    explain: list[str] = []

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

    # unique with stable order + resolve aliases
    seen = set()
    uniq: list[str] = []
    for variant in out:
        canonical = _resolve_alias(variant)
        if canonical not in seen:
            uniq.append(canonical)
            seen.add(canonical)

    if not explain:
        explain.append("variants задан списком напрямую")

    return uniq, "\n".join(explain)
