"""S1.7 sandbox: the runner refuses to start without its isolation flags (no Docker needed)."""

import pytest

from router.audit_sandbox import runner

SPEC = runner.Spec(image="example/image@sha256:" + "0" * 64)


def args():
    return runner.docker_args(SPEC, "s17-test")


def test_default_args_pass():
    runner.check_args(args())


@pytest.mark.parametrize("drop", [("--network", "none"), ("--read-only",), ("--user", runner.USER),
                                  ("--cpus", "1.0"), ("--memory", "2g"), ("--pids-limit", "256"),
                                  ("--cap-drop", "ALL"), ("--security-opt", "no-new-privileges")])
def test_refuses_without_each_flag(drop):
    a = args()
    i = a.index(drop[0])
    del a[i:i + len(drop)]
    with pytest.raises(PermissionError):
        runner.check_args(a)


@pytest.mark.parametrize("extra", [["-v", "/:/host"], ["--mount", "type=bind,src=/,dst=/h"],
                                   ["-e", "SECRET=1"], ["--privileged"]])
def test_refuses_host_exposure(extra):
    a = args()
    a[2:2] = extra
    with pytest.raises(PermissionError):
        runner.check_args(a)


def test_refuses_network_and_tag_images():
    a = args()
    a[a.index("--network") + 1] = "bridge"
    with pytest.raises(PermissionError):
        runner.check_args(a)
    b = runner.docker_args(runner.Spec(image="python:3.11"), "x")
    with pytest.raises(PermissionError):
        runner.check_args(b)


def test_accepts_local_image_id_only_in_full():
    runner.check_args(runner.docker_args(runner.Spec(image="sha256:" + "a" * 64), "x"))
    with pytest.raises(PermissionError):
        runner.check_args(runner.docker_args(runner.Spec(image="sha256:abc"), "x"))


def test_mx_specs_pass_the_isolation_check():
    """Checkpoint 2b (Amendment 9): every toolchain spec keeps the sandbox flags."""
    from router.experiments.s1_7_cp2 import MX_SPECS
    for lang, spec in MX_SPECS.items():
        if spec.image == "unbuilt":
            continue
        args = runner.docker_args(spec, f"t-{lang}")
        runner.check_args(args)
        assert "--network" in args and args[args.index("--network") + 1] == "none"
        assert not any(k in ("GOPROXY",) and v != "off" for k, v in spec.env)
