# Stage Qwen policy training

Use the official post-trained Qwen3.5-4B checkpoint as the Student Policy and train language-layer LoRA adapters with Unsloth. Prefer BF16 when the local VRAM feasibility test passes; any quantized fallback is an explicitly exploratory result because quantization may confound policy and reasoning quality.

Training has two stages. Teacher-guided GRPO first supplies dense action-quality rewards from the Depth-2 Teacher Policy without supervising the Student Policy's reasoning text. Environment GRPO then optimizes actual 2048 outcomes with no teacher action reward. A Reasoning Policy and a Direct-action Policy are trained as controlled alternatives, and promotion is decided primarily by 2048 Success Rate rather than Teacher Action Agreement or the apparent quality of generated reasoning.

Do not implement the GRPO optimizer, advantage calculation, or training loop from scratch. Teacher-guided training uses Unsloth with the maintained TRL `GRPOTrainer`; project-specific code is limited to the 2048 environment boundary, data flow, rewards, parsing, evaluation, and telemetry. The multi-turn Environment GRPO backend is deliberately deferred until a throwaway prototype compares Unsloth with ART against TRL `environment_factory` on the local hardware.
