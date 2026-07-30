"""Prompts and response contracts for the two Student Policy variants."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import cast, Literal, Sequence


Action = Literal["LEFT", "RIGHT", "UP", "DOWN"]
PolicyVariant = Literal["direct_action", "reasoning"]

ACTIONS: frozenset[str] = frozenset({"LEFT", "RIGHT", "UP", "DOWN"})
ACTION_ENVELOPE = "<action>ACTION</action>"
REASONING_ENVELOPE = (
    "<think>POLICY_REASONING_TRACE</think><action>ACTION</action>"
)
REASONING_MAX_GENERATION_TOKENS = 96

_ACTION_PATTERN = re.compile(r"<action>(.*?)</action>", re.DOTALL)
_DIRECT_ACTION_PATTERN = re.compile(
    r"<action>(LEFT|RIGHT|UP|DOWN)</action>"
)
_REASONING_PATTERN = re.compile(
    r"<think>(?P<trace>[^<]+)</think>"
    r"<action>(?P<action>LEFT|RIGHT|UP|DOWN)</action>"
)


@dataclass(frozen=True)
class PolicyContractResult:
    action: Action | None
    parsed: bool
    policy_failure_reason: str | None
    policy_reasoning_trace: str | None
    valid_action: bool

    @property
    def policy_failure(self) -> bool:
        return self.policy_failure_reason is not None


def build_policy_prompt(
    variant: PolicyVariant,
    board: Sequence[Sequence[int]],
) -> str:
    """Build the fixed Markov Policy prompt containing only the latest board."""
    compact_board = json.dumps(board, separators=(",", ":"))
    if variant == "direct_action":
        return (
            "You are the Direct-action Policy for 2048.\n"
            "Board (4x4 JSON array; 0 means empty):\n"
            f"{compact_board}\n"
            f"Return exactly one action in this form: {ACTION_ENVELOPE}."
        )
    return (
        "You are the Reasoning Policy for 2048.\n"
        "Board (4x4 JSON array; 0 means empty):\n"
        f"{compact_board}\n"
        "Write a concise English Policy Reasoning Trace and then exactly one "
        f"action in this form: {REASONING_ENVELOPE}."
    )


def enforce_policy_response(
    *,
    variant: PolicyVariant,
    response: str,
    truncated: bool,
    legal_actions: Sequence[str],
) -> PolicyContractResult:
    """Parse one response without retrying, repairing, or masking its action."""
    if truncated:
        return _failure("truncated_response")

    action_envelopes = _ACTION_PATTERN.findall(response)
    action_openings = response.count("<action>")
    if action_openings > 1 or len(action_envelopes) > 1:
        return _failure("multiple_actions")
    if not action_envelopes:
        if "<action" in response or "</action>" in response:
            return _failure("malformed_response")
        return _failure("missing_action")

    action_text = action_envelopes[0]
    if action_text not in ACTIONS:
        return _failure("out_of_vocabulary_action")

    reasoning_trace: str | None = None
    if variant == "direct_action":
        contract_match = _DIRECT_ACTION_PATTERN.fullmatch(response)
    else:
        contract_match = _REASONING_PATTERN.fullmatch(response)
        if contract_match is not None:
            reasoning_trace = contract_match.group("trace")
            if re.search(r"[A-Za-z]", reasoning_trace) is None:
                contract_match = None

    if contract_match is None:
        return _failure("malformed_response")

    action = cast(Action, action_text)
    if action not in legal_actions:
        return PolicyContractResult(
            action=action,
            parsed=True,
            policy_failure_reason="illegal_action",
            policy_reasoning_trace=reasoning_trace,
            valid_action=False,
        )
    return PolicyContractResult(
        action=action,
        parsed=True,
        policy_failure_reason=None,
        policy_reasoning_trace=reasoning_trace,
        valid_action=True,
    )


def _failure(reason: str) -> PolicyContractResult:
    return PolicyContractResult(
        action=None,
        parsed=False,
        policy_failure_reason=reason,
        policy_reasoning_trace=None,
        valid_action=False,
    )
