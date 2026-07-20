# llmcall

One lightweight primitive for every headless codex -> cc -> claude text-judgment call: a cost/health
ordered provider chain (codex first, then cc, then claude), read-only and one-shot, behind one small
API and CLI. It replaces four independently drifting in-house implementations.

Status: design approved 2026-07-19. See `the design notes` (design)
and `the inventory notes` (why). Implementation follows the writing-plans step.

Every consumer of this package (consumers) is tracked in `the docs`.
