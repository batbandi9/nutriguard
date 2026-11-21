"""ADK callbacks: the security layer around every model call and tool call.

1. input_guardrail      (before_model_callback) -> blocks prompt injection
2. tool_policy_guardrail (before_tool_callback) -> least privilege + policy-as-code
3. audit_after_tool      (after_tool_callback)  -> append-only audit log
"""

from __future__ import annotations

from typing import Any, Optional

from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from . import policy
from .audit import log_event
from .catalog import find_product


def _tag(ctx) -> str:
    return ctx.state.get("session_tag") or ctx.session.id


def _user_text(callback_context: CallbackContext) -> str:
    content = callback_context.user_content
    if not content or not content.parts:
        return ""
    return " ".join(p.text for p in content.parts if getattr(p, "text", None))


# ------------------------------------------------------------ 0. setup -----
def tag_session(callback_context: CallbackContext) -> None:
    """Give every conversation a stable tag so nested agent runs share one audit trail."""
    if not callback_context.state.get("session_tag"):
        callback_context.state["session_tag"] = callback_context.session.id
    return None


# ------------------------------------------------------------ 1. input -----
def input_guardrail(callback_context: CallbackContext,
                    llm_request: LlmRequest) -> Optional[LlmResponse]:
    """Stop prompt injection and out-of-scope medical requests before the LLM."""
    text = _user_text(callback_context)

    med = policy.detect_medical_request(text)
    if med:
        from .tools import summarize_cart  # local import avoids a cycle
        cart = summarize_cart(callback_context.state)
        log_event("medical_boundary", session=_tag(callback_context),
                  agent=callback_context.agent_name, matched=med, input=text[:300])
        gl_line = (f"Your current cart has an **estimated glycemic load of "
                   f"{cart['total_glycemic_load']}** across {cart['total_carbs_g']} g of carbs "
                   f"and {cart['total_kcal']} kcal in total. Lower is gentler on blood sugar."
                   if cart["items"] else
                   "Once you have items in your cart, I can show their **estimated glycemic "
                   "load**, a measure of how strongly food raises blood sugar.")
        return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=(
            "🩺 **This is outside what NutriGuard is allowed to do.**\n\n"
            "- **Blood glucose** can't be calculated from food. It depends on your body, "
            "medication, activity and timing, and must be **measured** with a glucometer or CGM.\n"
            "- **Insulin/medication doses and personal calorie targets** must come from your "
            "doctor or a registered dietitian. Cutting calories while on glucose-lowering "
            "medication can cause dangerous hypoglycemia.\n\n"
            f"What I *can* do: {gl_line} I can also show which foods are recommended, "
            "limited or avoided for your profile."
        ))]))

    hit = policy.detect_injection(text)
    if not hit:
        return None

    blocks = int(callback_context.state.get("security_blocks", 0)) + 1
    callback_context.state["security_blocks"] = blocks
    log_event("prompt_injection_blocked", session=_tag(callback_context),
              agent=callback_context.agent_name, matched=hit, input=text[:300])
    return LlmResponse(content=types.Content(role="model", parts=[types.Part(text=(
        "🛡️ **Blocked by NutriGuard security policy.** Your message looks like an "
        f"attempt to override the safety rules (matched: \"{hit}\"). Sugar limits, "
        "allergens and the blacklist are enforced in code and cannot be changed "
        "by chat instructions. I'm happy to help you build a healthy list instead!"
    ))]))


# ------------------------------------------------------------- 2. tools -----
def tool_policy_guardrail(tool: BaseTool, args: dict[str, Any],
                          tool_context: ToolContext) -> Optional[dict]:
    """Zero-trust check before ANY tool runs. Return a dict to block it."""
    agent = tool_context.agent_name
    session = _tag(tool_context)

    # Least privilege: explicit allowlist of tools per agent.
    allowed = policy.TOOL_PERMISSIONS.get(agent, set())
    if tool.name not in allowed:
        log_event("tool_blocked", session=session, agent=agent, tool=tool.name,
                  args=args, reason="least-privilege: tool not permitted for agent")
        return {"status": "blocked_by_policy",
                "reason": f"Agent '{agent}' is not permitted to call '{tool.name}'."}

    if tool.name == "add_to_list":
        product = find_product(str(args.get("product_name", "")))
        try:
            qty = int(args.get("quantity", 1))
        except (TypeError, ValueError):
            qty = -1
        profile = policy.get_profile(tool_context.state)
        cart = policy.get_cart(tool_context.state)
        violations = policy.evaluate_item(product, qty, profile, cart)
        if violations:
            alts = policy.suggest_alternatives(product, profile, cart)
            log_event("tool_blocked", session=session, agent=agent, tool=tool.name,
                      args=args, reason="; ".join(violations))
            return {
                "status": "blocked_by_policy",
                "product": product["name"] if product else args.get("product_name"),
                "policy_violations": violations,
                "suggested_alternatives": alts,
                "note": "This decision is final and enforced in code. Do not retry; "
                        "offer one of the suggested alternatives instead.",
            }

    if tool.name == "update_profile":
        sugar = args.get("sugar_limit_g_per_100g")
        budget = args.get("budget_eur")
        problems = []
        if sugar is not None and not (0 < float(sugar) <= policy.SYSTEM_MAX_SUGAR_PER_100G):
            problems.append(f"sugar limit must be >0 and ≤ {policy.SYSTEM_MAX_SUGAR_PER_100G} "
                            "g/100g (it can only be tightened)")
        if budget is not None and not (0 < float(budget) <= policy.MAX_BUDGET_EUR):
            problems.append(f"budget must be between €1 and €{policy.MAX_BUDGET_EUR:.0f}")
        if problems:
            log_event("tool_blocked", session=session, agent=agent, tool=tool.name,
                      args=args, reason="; ".join(problems))
            return {"status": "blocked_by_policy", "policy_violations": problems}

    log_event("tool_allowed", session=session, agent=agent, tool=tool.name, args=args)
    return None


# ------------------------------------------------------------- 3. audit -----
def audit_after_tool(tool: BaseTool, args: dict[str, Any], tool_context: ToolContext,
                     tool_response: Any) -> Optional[dict]:
    status = tool_response.get("status") if isinstance(tool_response, dict) else "ok"
    log_event("tool_result", session=_tag(tool_context), agent=tool_context.agent_name,
              tool=tool.name, status=status)
    return None
