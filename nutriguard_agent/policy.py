"""Policy-as-code: the rules NutriGuard enforces regardless of what the LLM says.

"The LLM suggests, the code decides."
All limits live here so judges / auditors can review them in one place.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .catalog import find_product, load_catalog

# ---------------------------------------------------------------- limits ---
SYSTEM_MAX_SUGAR_PER_100G = 10.0  # hard ceiling; users may only tighten it
SUGAR_EXEMPT_CATEGORIES = {"fruit", "vegetable"}  # natural sugars, whole foods
MAX_QTY_PER_ITEM = 6
MAX_BUDGET_EUR = 200.0
MAX_CART_LINES = 25

# Default-deny: only these categories may ever enter the cart (WHITELIST).
ALLOWED_CATEGORIES = {
    "protein_meat", "fish", "eggs", "dairy", "tofu_soy", "legume",
    "grain_whole", "vegetable", "fruit", "nuts_seeds",
    "protein_supplement", "beverage_unsweetened", "fat_oil",
    "grain_refined", "vegetable_starchy", "dairy_lactose_free",
}
# Explicitly banned categories (BLACKLIST), with a human-readable reason.
BLOCKED_CATEGORIES = {
    "soda": "sugary soft drinks are banned",
    "candy": "candy is banned",
    "sweet_spread": "sweet spreads are banned",
    "energy_drink": "energy drinks are banned",
    "juice": "fruit juice is banned (liquid sugar)",
    "breakfast_cereal_sugary": "sugary cereals are banned",
    "alcohol": "alcohol is banned",
    "pastry": "pastries are banned",
}

DEFAULT_PROFILE = {
    "goal": "high protein, low sugar",
    "sugar_limit_g_per_100g": SYSTEM_MAX_SUGAR_PER_100G,
    "allergens": [],
    "conditions": [],
    "budget_eur": 60.0,
}

# ------------------------------------------------- health conditions -------
# Client-dependent rules. Several conditions combine; the strictest rule wins.
# Thresholds follow common dietetic guidance (GI: low <=55, medium 56-69, high >=70).
SUPPORTED_CONDITIONS = {
    "diabetes": "Diabetes (type 1/2): avoid high-GI and sugary foods, watch carb portions.",
    "prediabetes": "Prediabetes: same food rules as diabetes, focus on low-GI foods.",
    "weight_management": "Overweight / weight management: flag energy-dense foods. "
                         "NutriGuard never prescribes calorie cuts.",
    "celiac": "Celiac disease: gluten is a hard block.",
    "lactose_intolerance": "Lactose intolerance: regular dairy is a hard block.",
}
CONDITION_SYNONYMS = {
    "diabetes": "diabetes", "diabetic": "diabetes", "type 1 diabetes": "diabetes",
    "type 2 diabetes": "diabetes", "t1d": "diabetes", "t2d": "diabetes",
    "type 2": "diabetes", "type 1": "diabetes", "sugar disease": "diabetes",
    "prediabetes": "prediabetes", "pre-diabetes": "prediabetes",
    "insulin resistance": "prediabetes",
    "overweight": "weight_management", "obesity": "weight_management",
    "obese": "weight_management", "fat": "weight_management",
    "weight loss": "weight_management", "weight management": "weight_management",
    "lose weight": "weight_management",
    "celiac": "celiac", "coeliac": "celiac", "celiac disease": "celiac",
    "lactose intolerance": "lactose_intolerance", "lactose intolerant": "lactose_intolerance",
    "lactose": "lactose_intolerance",
}
CONDITION_ALLERGENS = {"celiac": ["gluten"], "lactose_intolerance": ["milk"]}
GLUCOSE_CONDITIONS = {"diabetes", "prediabetes"}
GLUCOSE_SUGAR_LIMIT = 5.0      # g/100g for non-fruit/veg when glucose control matters
HIGH_GI = 70
MEDIUM_GI = 56
HIGH_GL_PER_100G = 20           # glycemic load per 100 g -> portion warning
ENERGY_DENSE_KCAL = 400         # kcal/100g -> portion warning for weight management

MEDICAL_BOUNDARY_NOTE = (
    "NutriGuard does not calculate blood glucose, insulin/medication doses or "
    "personal calorie targets. Blood glucose must be measured (glucometer/CGM). "
    "Calorie restriction for people with diabetes, especially on insulin or "
    "sulfonylureas, can cause hypoglycemia, so targets must come from a doctor "
    "or registered dietitian.")

# Least privilege / zero trust: which agent may call which tool.
TOOL_PERMISSIONS: dict[str, set[str]] = {
    "nutriguard_coordinator": {
        "get_profile", "update_profile", "nutrition_analyst", "list_builder",
        "view_list",  # read-only access to exact totals (no write tools)
    },
    "nutrition_analyst": {"lookup_nutrition", "find_alternatives",
                          "get_food_guidance"},  # read-only
    "list_builder": {"add_to_list", "remove_from_list", "view_list"},  # only writer
}

ALLERGEN_SYNONYMS = {
    "nut": ["tree_nuts", "peanuts"], "nuts": ["tree_nuts", "peanuts"],
    "tree nuts": ["tree_nuts"], "tree_nuts": ["tree_nuts"],
    "hazelnut": ["tree_nuts"], "almond": ["tree_nuts"],
    "peanut": ["peanuts"], "peanuts": ["peanuts"],
    "dairy": ["milk"], "lactose": ["milk"], "milk": ["milk"],
    "gluten": ["gluten"], "wheat": ["gluten"],
    "soy": ["soy"], "soya": ["soy"],
    "egg": ["eggs"], "eggs": ["eggs"],
    "fish": ["fish"], "seafood": ["fish"],
}

# Prompt-injection signatures (user input and untrusted data).
INJECTION_PATTERNS = [
    r"ignore (all |any |the |your )?(previous |prior |above )?(rules|instructions|policy|policies|limits)",
    r"(disable|bypass|turn off|override|remove) (the )?(sugar |safety |nutrition )?(rules?|policy|policies|limits?|filters?|guardrails?)",
    r"\b(i am|i'm) (the |an )?(admin|administrator|developer|system)\b",
    r"pre-?approved by (the )?admin",
    r"\bsystem (note|prompt|override|message)\b",
    r"\byou are now\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
]
_INJECTION_RE = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

# Medical-boundary requests: answered deterministically, never by the LLM.
MEDICAL_PATTERNS = [
    r"(calculate|predict|estimate|compute|measure|what('s| is)|tell me) (my |the )?(current )?(blood (sugar|glucose)|glucose level|sugar level|hba1c)",
    r"\b(insulin|metformin|medication|medicine) (dose|dosage|units|amount)\b",
    r"how (much|many units of) insulin",
    r"how many (calories|kcal) (should|do|must) i (eat|have|cut)",
    r"\b(calorie|caloric|kcal) (deficit|target|goal) for me\b",
    r"\bcut (my )?calories\b",
]
_MEDICAL_RE = [re.compile(p, re.IGNORECASE) for p in MEDICAL_PATTERNS]


def detect_medical_request(text: str) -> Optional[str]:
    if not text:
        return None
    for rx in _MEDICAL_RE:
        m = rx.search(text)
        if m:
            return m.group(0)
    return None


# --------------------------------------------------------------- helpers ---
def normalize_allergens(items: list[str]) -> list[str]:
    out: list[str] = []
    for raw in items or []:
        key = raw.strip().lower()
        out.extend(ALLERGEN_SYNONYMS.get(key, [key.replace(" ", "_")]))
    return sorted(set(out))


def normalize_conditions(items: list[str]) -> tuple[list[str], list[str]]:
    """Map free text to supported conditions. Returns (known, unknown)."""
    known, unknown = set(), []
    for raw in items or []:
        key = raw.strip().lower().replace("_", " ")
        cond = CONDITION_SYNONYMS.get(key) or (key.replace(" ", "_")
                                               if key.replace(" ", "_") in SUPPORTED_CONDITIONS else None)
        if cond:
            known.add(cond)
        else:
            unknown.append(raw)
    return sorted(known), unknown


def get_profile(state: Mapping[str, Any]) -> dict:
    profile = dict(DEFAULT_PROFILE)
    profile.update(state.get("profile") or {})
    conds = set(profile.get("conditions") or [])
    allergens = set(profile.get("allergens") or [])
    for c in conds:
        allergens |= set(CONDITION_ALLERGENS.get(c, []))
    profile["allergens"] = sorted(allergens)
    if conds & GLUCOSE_CONDITIONS:
        profile["sugar_limit_g_per_100g"] = min(
            float(profile["sugar_limit_g_per_100g"]), GLUCOSE_SUGAR_LIMIT)
    return profile


def glycemic_load(product: Mapping[str, Any], grams: float = 100) -> float:
    """GL = GI x available carbs (g) / 100. Estimates the blood-sugar impact of a portion."""
    return round(product.get("gi", 0) * product.get("carbs_100g", 0) * grams / 100 / 100, 1)


def classify_item(product: Optional[dict], profile: Mapping[str, Any]) -> dict:
    """Client-dependent verdict: 'avoid' (hard block), 'limit' (warn) or 'recommended'."""
    if product is None:
        return {"verdict": "avoid", "avoid": ["unknown product (default deny)"], "limit": []}
    conds = set(profile.get("conditions") or [])
    cat = product["category"]
    avoid: list[str] = []
    limit: list[str] = []

    if cat in BLOCKED_CATEGORIES:
        avoid.append(f"BLACKLIST: {BLOCKED_CATEGORIES[cat]}")
    elif cat not in ALLOWED_CATEGORIES:
        avoid.append(f"category '{cat}' is not on the whitelist (default deny)")

    sugar_limit = min(float(profile.get("sugar_limit_g_per_100g", SYSTEM_MAX_SUGAR_PER_100G)),
                      SYSTEM_MAX_SUGAR_PER_100G)
    if cat not in SUGAR_EXEMPT_CATEGORIES and product["sugar_100g"] > sugar_limit:
        avoid.append(f"SUGAR: {product['sugar_100g']} g/100g exceeds limit of {sugar_limit} g/100g")

    hits = sorted(set(product.get("allergens", [])) & set(profile.get("allergens", [])))
    if hits:
        avoid.append(f"ALLERGEN/INTOLERANCE: contains {', '.join(hits)}")

    gi = product.get("gi", 0)
    gl = glycemic_load(product)
    if conds & GLUCOSE_CONDITIONS:
        if gi >= HIGH_GI:
            avoid.append(f"DIABETES: high glycemic index (GI {gi}) spikes blood sugar")
        elif gi >= MEDIUM_GI:
            limit.append(f"DIABETES: medium GI ({gi}), small portions, pair with protein")
        if cat == "fruit" and product["sugar_100g"] > 10:
            limit.append(f"DIABETES: fruit with {product['sugar_100g']} g sugar/100g, max 1 portion per meal")
        if gl >= HIGH_GL_PER_100G and gi < HIGH_GI:
            limit.append(f"DIABETES: glycemic load {gl} per 100 g, keep portions to ~40-50 g")
    if "weight_management" in conds and product["kcal_100g"] >= ENERGY_DENSE_KCAL:
        limit.append(f"WEIGHT: energy-dense ({product['kcal_100g']} kcal/100g), use small portions")

    verdict = "avoid" if avoid else ("limit" if limit else "recommended")
    return {"verdict": verdict, "avoid": avoid, "limit": limit,
            "gi": gi, "glycemic_load_per_100g": gl}


def get_cart(state: Mapping[str, Any]) -> dict[str, int]:
    return dict(state.get("cart") or {})


def cart_cost(cart: Mapping[str, int]) -> float:
    total = 0.0
    for name, qty in cart.items():
        p = find_product(name)
        if p:
            total += p["price_eur"] * qty
    return round(total, 2)


def detect_injection(text: str) -> Optional[str]:
    """Return the matched snippet if text looks like a prompt injection."""
    if not text:
        return None
    for rx in _INJECTION_RE:
        m = rx.search(text)
        if m:
            return m.group(0)
    return None


def sanitize_untrusted_text(text: str) -> tuple[str, bool]:
    """Neutralise instructions hidden in third-party data (e.g. product text)."""
    if detect_injection(text):
        return ("[REDACTED: this product description contained instructions "
                "aimed at the AI and was removed by NutriGuard]"), True
    return text, False


def evaluate_item(product: Optional[dict], qty: int,
                  profile: Mapping[str, Any], cart: Mapping[str, int]) -> list[str]:
    """Return a list of policy violations (empty list == allowed)."""
    if product is None:
        return ["unknown product: not in the verified catalog (default deny)"]

    reasons: list[str] = list(classify_item(product, profile)["avoid"])

    if not isinstance(qty, int) or qty < 1 or qty > MAX_QTY_PER_ITEM:
        reasons.append(f"QUANTITY: must be between 1 and {MAX_QTY_PER_ITEM}")
    else:
        new_cart = dict(cart)
        new_cart[product["name"]] = new_cart.get(product["name"], 0) + qty
        if new_cart[product["name"]] > MAX_QTY_PER_ITEM:
            reasons.append(f"QUANTITY: max {MAX_QTY_PER_ITEM} packs per item")
        if len(new_cart) > MAX_CART_LINES:
            reasons.append(f"CART: max {MAX_CART_LINES} different items")
        budget = min(float(profile.get("budget_eur", MAX_BUDGET_EUR)), MAX_BUDGET_EUR)
        if cart_cost(new_cart) > budget:
            reasons.append(
                f"BUDGET: cart would cost €{cart_cost(new_cart):.2f}, budget is €{budget:.2f}")
    return reasons


def suggest_alternatives(product: Optional[dict], profile: Mapping[str, Any],
                         cart: Mapping[str, int], k: int = 3) -> list[dict]:
    """Deterministic, policy-compliant swaps ranked by protein."""
    same_cat = product["category"] if product else None
    candidates = []
    for p in load_catalog():
        if product and p["name"] == product["name"]:
            continue
        if evaluate_item(p, 1, profile, cart):
            continue
        same_family = same_cat and p["category"].split("_")[0] == same_cat.split("_")[0]
        score = min(p["protein_100g"], 30) + (15 if same_family else 0)
        if classify_item(p, profile)["verdict"] == "limit":
            score -= 12  # prefer foods that are fully recommended for this client
        if product and product["category"] in {"sweet_spread", "breakfast_cereal_sugary", "candy"}:
            score += 10 if p["category"] in {"dairy", "nuts_seeds", "grain_whole", "fruit"} else 0
        if product and product["category"] in {"soda", "juice", "energy_drink"}:
            score += 50 if p["category"] == "beverage_unsweetened" else 0
        candidates.append((score, p))
    candidates.sort(key=lambda t: -t[0])
    return [
        {"name": p["name"], "protein_100g": p["protein_100g"],
         "sugar_100g": p["sugar_100g"], "price_eur": p["price_eur"]}
        for _, p in candidates[:k]
    ]
