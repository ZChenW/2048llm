# First paired Teacher-guided GRPO block

Date: 2026-07-31

This report records the bounded experiment from issue #9. It does not override
the failed issue-8 gate and does not claim Project Success.

## Registered conditions

Both policies used Qwen3.5-4B revision
`c7429d5a8ed57f4a9cfdaf1af76a8943eba0ae97`, BF16 language-only rank-64
LoRA, maintained TRL `GRPOTrainer`, group size 4, temperature 1.0, top-p
0.95, top-k 20, and a 96-token completion limit. Each policy completed 250
optimizer steps against Dynamic Board Pool generation 0:

- shared pool identity:
  `fef407cce17ab2948ded92d4d8ec852fbe0447f85424a2d9db978c6c52e71829`
- immutable 70,000-board Teacher Core member identity:
  `cb4e7fb0b81fbeab39fa5ced138ee098568f82c61156ebaa4b576e321169a42f`
- fixed validation snapshot identity:
  `66f981c54d7c0aea7da072627cf1d232029317eecbb9672a0180f510964881ba`
- source gate evidence:
  `f4f85c85c9542e9d6a45ecd4681362530b210ca2c4d33343a816fca3233c10fc`
  with `ready_for_teacher_guided_grpo=false`

The full 70,000-board prompt audit passed for both policies. Direct-action
prompts used 114–135 tokens and Reasoning prompts used 126–147 tokens, all
below the registered 160-token limit.

## Results

| Metric | Direct-action | Reasoning |
| --- | ---: | ---: |
| Optimizer steps | 250 | 250 |
| Initialization mean reward | 0.259125 | -1.250000 |
| Step-250 mean reward | 0.262023 | -1.250000 |
| Paired mean reward delta | +0.002898 | 0.000000 |
| Paired 95% bootstrap CI | [-0.009746, 0.015530] | [0.000000, 0.000000] |
| Measurable learning | no | no |
| Initialization Policy Failure | 8.80% | 100.00% |
| Step-250 Policy Failure | 7.65% | 100.00% |
| Initialization Teacher Agreement | 22.25% | 0.00% |
| Step-250 Teacher Agreement | 21.75% | 0.00% |
| Initialization action entropy (nats) | 0.341256 | 0.000000 |
| Step-250 action entropy (nats) | 0.000000 | 0.000000 |

Direct-action collapsed from 1,785 LEFT / 215 RIGHT initialization actions to
2,000 LEFT actions at step 250. Its reward improvement confidence interval
crosses zero, Teacher Agreement fell, and the final illegal-action rate
remained far above the registered 2% gate. The training ledger contains 1,000
reward events; 176 of 250 optimizer steps had zero reward dispersion and zero
gradient, including every step from 115 through 250. This is mode collapse,
not measurable learning.

Reasoning emitted an empty trace followed by an action,
`<think>\n</think><action>...</action>`. Every initialization, training, and
step-250 response therefore failed the strict trace contract as
`non_english_reasoning_trace`. Although all four action tags appeared during
stochastic training (DOWN 366, LEFT 282, RIGHT 185, UP 167), all 1,000
candidates received -1.25. Consequently all 250 Rollout Groups had zero reward
dispersion, loss 0, and gradient norm 0. The step-250 adapter hash exactly
equals both the step-125 and starting adapter hashes:
`65ec5ed5b77ccb2de24ddb08f0d28ec4b752b35915ccaaab3dedd9c5cecacd4c`.

## Resume and checkpoint evidence

Direct-action was stopped at step 125 and resumed in a fresh process from
`trainer/checkpoint-125`. The final result records source optimizer step 125,
500 prior reward events, prior ledger SHA
`005af6593bca7cfdbcadea9df32c94b7f7d0514bd628c2698f83ba25e7547672`,
and resume-manifest SHA
`1b076703167944145e047616e8e787b48744643f9f07c1c09e72fb2e50207cc8`.
The resumed history is continuous through step 250 and retains exactly 1,000
reward events. Reasoning ran uninterrupted; it is not presented as a second
resume proof.

Both step-250 checkpoints contain the LoRA adapter, optimizer, scheduler, RNG,
trainer state, resolved configuration, pool identity, validation identity, and
resume manifest. The final adapter hashes are:

- Direct-action:
  `a9af4f8ce1c0b75d0b61a5d22701821be8b79c4d0e7338205b53f48016d7dde9`
- Reasoning:
  `65ec5ed5b77ccb2de24ddb08f0d28ec4b752b35915ccaaab3dedd9c5cecacd4c`

## Telemetry and immutable result identities

- Direct-action result SHA:
  `747ac12bb5716256a02218a31de1753619fc41ce08a517de8c2ed8ef43d0027a`
- Reasoning result SHA:
  `2148fbf0fde907acaff3d215a015b5f533316f544c1442762589e03225b4e66a`
- Reasoning resume-manifest SHA:
  `bbcfd9731bfaa3cbc21046e2ee9b54259b1a714c0fc5ac51921f78d4076ed0bc`
- [Direct-action private W&B run](https://wandb.ai/zichenw66-umass-amherst/2048llm-feasibility/runs/jh3qrke6)
- [Reasoning private W&B run](https://wandb.ai/zichenw66-umass-amherst/2048llm-feasibility/runs/k5ngkor7)

Both W&B runs finished with comparison and evaluation metrics and zero uploaded
artifacts. Local TensorBoard event files exist for both runs. Transformers
5.5 placed them below `trainer/runs/` instead of the deprecated configured
`telemetry/tensorboard` path; the runner now discovers and records the actual
event files.

## Decision

Neither policy demonstrated measurable learning. There is no promoted
Teacher-guided candidate and no Project Success claim. The shared Dynamic
Board Pool feedback loop in issue #10 is held: refreshing the pool and spending
another paired 250-step block cannot repair Direct-action's collapsed
within-group signal or Reasoning's uniformly failed response contract. A new
ticket must first restore reward diversity and pass a bounded format/legality
gate; issues #10, #11, and the promoted-adapter dependency of #13 remain
scientifically blocked by this result.
