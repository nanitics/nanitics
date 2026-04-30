"""Structural contract test for ``DockerSandbox._build_host_config``.

We test a private method — deliberately. The security-critical flags
the method assembles (``read_only``, ``security_opt``, ``pids_limit``,
``cap_drop``, ``mem_limit``, ``nano_cpus``, ``dns``, ``dns_search``)
are HostConfig *inputs* to the Docker daemon, not observables from
inside a running container. Gating their regression test on a running
Docker daemon would put the invariant behind a CI dependency
(``@pytest.mark.docker``) that is skipped by default, which is the
wrong posture for a security contract. Testing the builder directly
keeps the invariant in the always-on suite.
"""

from __future__ import annotations

from nanitics.safety.sandbox.docker import _PID_LIMIT, DockerSandbox
from nanitics.safety.sandbox.protocol import SandboxConfig


class TestBuildHostConfigNetworkIsolated:
    def test_contains_security_critical_flags(self) -> None:
        cfg = SandboxConfig(network_access=False)
        sandbox = DockerSandbox(cfg)
        host_config = sandbox._build_host_config()

        assert host_config["read_only"] is True
        assert host_config["security_opt"] == ["no-new-privileges"]
        assert host_config["pids_limit"] == _PID_LIMIT
        assert host_config["cap_drop"] == ["NET_RAW"]
        assert host_config["mem_limit"] == f"{cfg.memory_limit_mb}m"
        assert host_config["nano_cpus"] == int(cfg.cpu_count * 1e9)
        assert host_config["dns"] == ["127.0.0.1"]
        assert host_config["dns_search"] == [""]


class TestBuildHostConfigNetworkAllowed:
    def test_network_access_true_drops_network_restrictions(self) -> None:
        cfg = SandboxConfig(network_access=True)
        sandbox = DockerSandbox(cfg)
        host_config = sandbox._build_host_config()

        # Core isolation flags remain regardless of network_access.
        assert host_config["read_only"] is True
        assert host_config["security_opt"] == ["no-new-privileges"]
        assert host_config["pids_limit"] == _PID_LIMIT
        assert host_config["mem_limit"] == f"{cfg.memory_limit_mb}m"
        assert host_config["nano_cpus"] == int(cfg.cpu_count * 1e9)

        # Network restrictions are only applied when network_access=False.
        assert "cap_drop" not in host_config
        assert "dns" not in host_config
        assert "dns_search" not in host_config
