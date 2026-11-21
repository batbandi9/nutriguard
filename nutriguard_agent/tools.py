"""Tools available to NutriGuard agents. Each agent only gets the tools it needs."""

from __future__ import annotations

from typing import Optional

from google.adk.tools.tool_context import ToolContext

from . import policy
from .audit import log_event
from .catalog import find_product, load_catalog, unit_totals


def _session_tag(tool_context: ToolContext) -> str:
    return tool_context.state.get("session_tag") or tool_context.session.id


# ------------------------------------------------- nutrition_analyst (read) --
def lookup_nutrition(product_name: str, tool_context: ToolContext) -> dict:
    """Looks up verified nutrition facts for a grocery product.

    Args:
        product_name: Name of the product, e.g. "greek yogurt" or "nutella".

    Returns:
        dict with status and nutrition facts per 100 g, price, allergens and
        whether the product passes the user's current policy.
    """
    product = find_product(product_name)
    if not product:
        return {"status": "not_found",
                "message": f"'{product_name}' is not in the verified catalog."}

    description, flagged = policy.sanitize_untrusted_text(product["description"])
    if flagged:
        log_event("data_injection_redacted", session=_session_tag(tool_context),
                  agent=tool_context.agent_name, product=product["name"])

    profile = policy.get_profile(tool_context.state)
    verdict = policy.classify_item(product, profile)
    return {
        "status": "success",
        "name": product["name"],
        "category": product["category"],
        "sugar_g_per_100g": product["sugar_100g"],
        "protein_g_per_100g": product["protein_100g"],
        "carbs_g_per_100g": product.get("carbs_100g"),
        "kcal_per_100g": product["kcal_100g"],
        "glycemic_index": verdict["gi"],
        "glycemic_load_per_100g": verdict["glycemic_load_per_100g"],
        "pack_size_g": product["unit_g"],
        "price_eur": product["price_eur"],
        "allergens": product["allergens"],
        "description_untrusted": description,
        "verdict_for_this_user": verdict["verdict"],
        "avoid_reasons": verdict["avoid"],
        "limit_reasons": verdict["limit"],
    }


def get_food_guidance(tool_context: ToolContext) -> dict:
    """Personal food guidance: which foods are recommended, which to limit and
    which to avoid, based on the user's conditions, allergies and goals.

    Returns:
        dict with 'recommended', 'limit' and 'avoid' food lists plus condition notes.
    """
    profile = policy.get_profile(tool_context.state)
    groups: dict[str, list] = {"recommended": [], "limit": [], "avoid": []}
    for p in load_catalog():
        v = policy.classify_item(p, profile)
        entry = {"name": p["name"], "protein_100g": p["protein_100g"],
                 "kcal_100g": p["kcal_100g"], "gi": p.get("gi", 0)}
        if v["verdict"] != "recommended":
            entry["why"] = (v["avoid"] or v["limit"])[0]
        groups[v["verdict"]].append(entry)
    groups["recommended"].sort(key=lambda e: -e["protein_100g"])
    return {
        "status": "success",
        "profile": profile,
        "condition_notes": [policy.SUPPORTED_CONDITIONS[c] for c in profile["conditions"]
                            if c in policy.SUPPORTED_CONDITIONS],
        **groups,
        "medical_boundary": policy.MEDICAL_BOUNDARY_NOTE,
    }


def find_alternatives(product_name: str, tool_context: ToolContext) -> dict:
    """Finds healthier, policy-compliant alternatives for a product.

    Args:
        product_name: The product to replace, e.g. "nutella" or "cola".

    Returns:
        dict with up to 3 alternatives ranked by protein content.
    """
    product = find_product(product_name)
    profile = policy.get_profile(tool_context.state)
    cart = policy.get_cart(tool_context.state)
    alts = policy.suggest_alternatives(product, profile, cart)
    return {"status": "success", "replacing": product["name"] if product else product_name,
            "alternatives": alts}


# ---------------------------------------------------- coordinator (profile) --
def get_profile(tool_context: ToolContext) -> dict:
    """Returns the user's nutrition profile: goal, sugar limit, allergens, budget."""
    return {"status": "success", "profile": policy.get_profile(tool_context.state),
            "system_max_sugar_g_per_100g": policy.SYSTEM_MAX_SUGAR_PER_100G}


def update_profile(
    tool_context: ToolContext,
    goal: Optional[str] = None,
    sugar_limit_g_per_100g: Optional[float] = None,
    add_allergens: Optional[list[str]] = None,
    add_conditions: Optional[list[str]] = None,
    budget_eur: Optional[float] = None,
) -> dict:
    """Updates the user's nutrition profile. Only provide the fields that change.

    Args:
        goal: Free-text goal, e.g. "high protein breakfast for a week".
        sugar_limit_g_per_100g: Max sugar per 100 g (can only be lowered, never above the system max).
        add_allergens: Allergens to add, e.g. ["nuts", "peanuts"]. Allergens can only be added, never removed.
        add_conditions: Health conditions to add, e.g. ["type 2 diabetes", "overweight", "celiac", "lactose intolerance"]. Add-only.
        budget_eur: Total grocery budget in euros.

    Returns:
        dict with the updated profile.
    """
    profile = dict(policy.DEFAULT_PROFILE)
    profile.update(tool_context.state.get("profile") or {})
    unknown: list[str] = []
    if add_conditions:
        known, unknown = policy.normalize_conditions(add_conditions)
        profile["conditions"] = sorted(set(profile.get("conditions") or []) | set(known))
    if goal:
        profile["goal"] = goal[:200]
    if sugar_limit_g_per_100g is not None:
        profile["sugar_limit_g_per_100g"] = float(sugar_limit_g_per_100g)
    if add_allergens:
        profile["allergens"] = sorted(
            set(profile["allergens"]) | set(policy.normalize_allergens(add_allergens)))
    if budget_eur is not None:
        profile["budget_eur"] = float(budget_eur)
    tool_context.state["profile"] = profile
    result = {"status": "success", "profile": policy.get_profile(tool_context.state)}
    if unknown:
        result["unsupported_conditions"] = unknown
        result["note"] = ("These conditions have no specific rules yet; general "
                          "healthy-eating rules apply. Advise consulting a dietitian.")
    return result


# ----------------------------------------------------- list_builder (write) --
def add_to_list(product_name: str, quantity: int, tool_context: ToolContext) -> dict:
    """Adds a product to the grocery cart. Every call is checked by the policy engine.

    Args:
        product_name: Product to add, e.g. "skyr".
        quantity: Number of packs (1-6).

    Returns:
        dict with status and the updated cart line.
    """
    product = find_product(product_name)
    if not product:  # defence in depth; the policy callback already checks
        return {"status": "error", "message": f"'{product_name}' not in catalog."}
    cart = policy.get_cart(tool_context.state)
    cart[product["name"]] = cart.get(product["name"], 0) + int(quantity)
    tool_context.state["cart"] = cart
    verdict = policy.classify_item(product, policy.get_profile(tool_context.state))
    result = {"status": "success", "added": product["name"], "quantity": int(quantity),
              "cart_total_eur": policy.cart_cost(cart)}
    if verdict["limit"]:
        result["warnings"] = verdict["limit"]  # allowed, but tell the user to limit it
    return result


def remove_from_list(product_name: str, tool_context: ToolContext) -> dict:
    """Removes a product from the grocery cart.

    Args:
        product_name: Product to remove.

    Returns:
        dict with status.
    """
    product = find_product(product_name)
    cart = policy.get_cart(tool_context.state)
    name = product["name"] if product else product_name
    if name not in cart:
        return {"status": "error", "message": f"'{name}' is not in the cart."}
    del cart[name]
    tool_context.state["cart"] = cart
    return {"status": "success", "removed": name}


def view_list(tool_context: ToolContext) -> dict:
    """Shows the current grocery cart with protein, sugar and price totals."""
    return {"status": "success", **summarize_cart(tool_context.state)}


def summarize_cart(state) -> dict:
    cart = policy.get_cart(state)
    profile = policy.get_profile(state)
    lines, protein, sugar, price, kcal, carbs, gl = [], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for name, qty in cart.items():
        p = find_product(name)
        if not p:
            continue
        t = unit_totals(p, qty)
        grams = p["unit_g"] * qty
        item_kcal = round(p["kcal_100g"] * grams / 100)
        item_carbs = round(p.get("carbs_100g", 0) * grams / 100, 1)
        item_gl = policy.glycemic_load(p, grams)
        protein += t["protein_g"]
        sugar += t["sugar_g"]
        price += t["price_eur"]
        kcal += item_kcal
        carbs += item_carbs
        gl += item_gl
        lines.append({"item": name, "qty": qty, **t, "kcal": item_kcal,
                      "carbs_g": item_carbs, "glycemic_load": item_gl,
                      "verdict": policy.classify_item(p, profile)["verdict"]})
    return {
        "items": lines,
        "total_protein_g": round(protein, 1),
        "total_sugar_g": round(sugar, 1),
        "total_kcal": round(kcal),
        "total_carbs_g": round(carbs, 1),
        "total_glycemic_load": round(gl, 1),
        "protein_kcal_share_pct": round(protein * 4 / kcal * 100) if kcal else 0,
        "total_price_eur": round(price, 2),
        "budget_eur": profile["budget_eur"],
        "budget_left_eur": round(profile["budget_eur"] - price, 2),
    }
