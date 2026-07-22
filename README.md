# llmcall

One lightweight primitive for every headless codex -> cc -> claude text-judgment call: a cost/health
ordered provider chain (codex first, then cc, then claude), read-only and one-shot, behind one small
API and CLI. It replaces several independently drifting in-house implementations with one tested,
never-raises primitive.

`refine(prompt, max_depth=N, judge=None)` adds opt-in iterative deepening: generate, then an independent judge decides DONE or CONTINUE and the answer is regenerated until it converges or hits max_depth.
