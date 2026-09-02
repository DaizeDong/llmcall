# -*- coding: utf-8 -*-
"""The list of environment variables core.py consults, in a module both conftest and its control
can import by name.

WHY IT IS NOT IN conftest.py ANY MORE. It was, and test_hermetic_env.py reached it with
`from conftest import STEERING_VARS`. That works only while the FIRST conftest.py pytest imports
happens to be this one: pytest imports a conftest by its bare basename, so any other conftest.py
higher in the tree claims the module name `conftest` and the import resolves to the wrong file.
Adding the guards submodule did exactly that, since `guards` sorts before `tests`, and the whole
suite stopped collecting with an ImportError naming a file in the submodule.

The variable list is shared data, so it lives in a module with a name of its own. conftest.py
imports it from here, and so does the control that keeps conftest honest.
"""

# Keep this list in step with core.py: an entry missing here is a way for the host machine to
# change a test result without saying so.
STEERING_VARS = (
    "LLMCALL_CHAIN",          # which providers, in which order
    "LLMCALL_AGENT_RUNNER",   # the external agent runner for the cc/claude agentic path
    "LLMCALL_RELAY",          # notification relay
    "LLMCALL_LEDGER",         # ledger destination
)
