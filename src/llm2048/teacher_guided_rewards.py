"""Action Quality Reward callback for Teacher-guided Rollout Groups."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from llm2048.policy_contracts import (
    Action,
    PolicyContractResult,
    PolicyVariant,
    change_making_actions,
    enforce_policy_response,
)


@dataclass(frozen=True)
class TeacherGuidedCompletion:
    variant: PolicyVariant
    response: str
    truncated: bool


@dataclass(frozen=True)
class RewardComponents:
    action_quality: float
    best_action_bonus: float
    illegal_action_penalty: float
    policy_failure_penalty: float

    @property
    def total(self) -> float:
        return (
            self.action_quality
            + self.best_action_bonus
            + self.illegal_action_penalty
            + self.policy_failure_penalty
        )

    def resolved(self) -> dict[str, float]:
        return {
            "action_quality": self.action_quality,
            "best_action_bonus": self.best_action_bonus,
            "illegal_action_penalty": self.illegal_action_penalty,
            "policy_failure_penalty": self.policy_failure_penalty,
        }


@dataclass(frozen=True)
class TeacherGuidedReward:
    contract: PolicyContractResult
    selected_action_score: float | None
    teacher_top1_score: float
    regret: float | None
    components: RewardComponents

    @property
    def total(self) -> float:
        return self.components.total


def teacher_guided_reward_callback(
    *,
    board: Sequence[Sequence[int]],
    completions: Sequence[TeacherGuidedCompletion],
    group_size: int,
    teacher_action_scores: Mapping[Action, float | None],
    teacher_action: Action,
    tau: float,
) -> list[TeacherGuidedReward]:
    """Score one complete group from the shared board and Teacher judgment."""
    if len(completions) != group_size:
        raise ValueError("Rollout Group does not match its configured group size")

    board_change_actions = change_making_actions(board)
    top1_score = teacher_action_scores[teacher_action]
    if top1_score is None:
        raise ValueError("Teacher-best action must have a score")

    rewards: list[TeacherGuidedReward] = []
    for completion in completions:
        contract = enforce_policy_response(
            variant=completion.variant,
            response=completion.response,
            truncated=completion.truncated,
            board_change_actions=board_change_actions,
        )
        if contract.policy_failure:
            illegal_action = contract.policy_failure_reason == "illegal_action"
            rewards.append(
                TeacherGuidedReward(
                    contract=contract,
                    selected_action_score=None,
                    teacher_top1_score=top1_score,
                    regret=None,
                    components=RewardComponents(
                        action_quality=0.0,
                        best_action_bonus=0.0,
                        illegal_action_penalty=-1.0 if illegal_action else 0.0,
                        policy_failure_penalty=0.0 if illegal_action else -1.25,
                    ),
                )
            )
            continue
        if contract.action is None:
            raise ValueError("valid Policy Response is missing its action")
        selected_score = teacher_action_scores[contract.action]
        if selected_score is None:
            raise ValueError("valid legal action is missing its Teacher score")
        regret = top1_score - selected_score
        rewards.append(
            TeacherGuidedReward(
                contract=contract,
                selected_action_score=selected_score,
                teacher_top1_score=top1_score,
                regret=regret,
                components=RewardComponents(
                    action_quality=math.exp(-regret / tau),
                    best_action_bonus=(
                        0.1 if contract.action == teacher_action else 0.0
                    ),
                    illegal_action_penalty=0.0,
                    policy_failure_penalty=0.0,
                ),
            )
        )
    return rewards
