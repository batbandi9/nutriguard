"""Product catalog: deterministic nutrition data the policy layer trusts.

The LLM never supplies nutrition numbers. Every policy decision re-reads
values from this catalog, so a model cannot "talk" a product into the cart.
"""

from __future__ import annotations

import difflib
import json
from functools import lru_cache
from pathlib import Path
from typing import Optional

_CATALOG_PATH = Path(__file__).with_name("products.json")


@lru_cache(maxsize=1)
def load_catalog() -> tuple[dict, ...]:
    with _CATALOG_PATH.open(encoding="utf-8") as f:
        return tuple(json.load(f))


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").split())


def find_product(name: str) -> Optional[dict]:
    """Find a product by name or alias (exact, then substring, then fuzzy)."""
    if not name:
        return None
    query = _norm(name)
    catalog = load_catalog()

    index: dict[str, dict] = {}
    for p in catalog:
        index[_norm(p["name"])] = p
        for alias in p.get("aliases", []):
            index[_norm(alias)] = p

    if query in index:
        return index[query]
    for key, p in index.items():
        if query in key or key in query:
            return p
    match = difflib.get_close_matches(query, list(index), n=1, cutoff=0.75)
    return index[match[0]] if match else None


def unit_totals(product: dict, qty: int) -> dict:
    """Protein / sugar / price for `qty` packs of a product."""
    grams = product["unit_g"] * qty
    return {
        "protein_g": round(product["protein_100g"] * grams / 100, 1),
        "sugar_g": round(product["sugar_100g"] * grams / 100, 1),
        "price_eur": round(product["price_eur"] * qty, 2),
    }
