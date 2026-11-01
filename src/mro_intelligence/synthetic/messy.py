"""Deliberate catalog messiness.

Real distributor item masters are messy in specific, repeatable ways. The
generator corrupts clean descriptions so M1 normalization/dedup has realistic
material to work with. Corruption is stronger on C-class SKUs — long-tail
neglect is the documented pattern the platform exists to fix.
"""

from __future__ import annotations

import numpy as np

ABBREVIATIONS: dict[str, str] = {
    "UNIVERSAL": "UNIV",
    "FILTER": "FLT",
    "CAPACITOR": "CAP",
    "CONTACTOR": "CONT",
    "MOTOR": "MTR",
    "RELIEF": "RLF",
    "PREVENTER": "PRV",
    "ELEMENT": "ELEM",
    "TRANSFORMER": "XFMR",
    "STANDING": "STDG",
    "IGNITION": "IGN",
    "STAINLESS": "SS",
    "REFRIGERANT": "REFRIG",
}


def corrupt_description(text: str, rng: np.random.Generator, level: float) -> str:
    """Apply 0..3 random corruptions; ``level`` in [0, 1] scales the dose."""
    tokens = text.split()
    n_ops = int(np.clip(rng.poisson(level * 2), 0, 3))
    for _ in range(n_ops):
        if not tokens:
            break
        op = rng.integers(0, 5)
        i = int(rng.integers(0, len(tokens)))
        if op == 0:  # abbreviate a token
            tokens[i] = ABBREVIATIONS.get(tokens[i].upper(), tokens[i])
        elif op == 1:  # case noise
            tokens = [t.lower() if rng.random() < 0.5 else t for t in tokens]
        elif op == 2:  # drop a token (attribute loss)
            if len(tokens) > 3:
                tokens.pop(int(rng.integers(0, len(tokens))))
        elif op == 3:  # transpose adjacent tokens
            if len(tokens) > 1:
                j = min(i + 1, len(tokens) - 1)
                tokens[i], tokens[j] = tokens[j], tokens[i]
        else:  # whitespace noise
            tokens[i] = f" {tokens[i]} "
    return " ".join(
        t.strip() if idx == len(tokens) - 1 else t for idx, t in enumerate(tokens)
    ).strip()


def corrupt_mpn(mpn: str, rng: np.random.Generator) -> str:
    """Supplier-code variant of the same MPN (duplicate-SKU signal)."""
    style = int(rng.integers(0, 3))
    if style == 0:
        return mpn.replace("-", "")
    if style == 1:
        return mpn + "-A"
    return mpn.lower()
