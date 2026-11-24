"""The only code that imports AgentDojo. It runs in a temporary uv environment (make dojo-source, make ground-truth).

Nothing else imports this package, and pyright skips it (pyproject.toml).
"""
