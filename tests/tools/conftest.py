"""Shared scaffolding for reference-tool tests.

Intentionally empty — the reference-tool test suites (``test_web_search``,
``test_http``, ``test_file_read``, ``test_code_execution``) rely on pytest's
standard fixtures (``tmp_path``, ``monkeypatch``) and ``respx.mock``; no
shared helpers are needed.  The file stays as a pytest discovery anchor.
"""
