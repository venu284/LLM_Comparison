"""S1.7 checkpoint 2c helper, run with an interpreter that has swebench 4.0.4 installed (kept
outside the project venv). Reads the pinned SWE-bench Lite extract.

    <py> swe_tool.py spec  <extract.jsonl> <instance_id>            -> JSON {eval_script, image_key}
    <py> swe_tool.py grade <extract.jsonl> <instance_id> <log> <patch> -> JSON {resolved, report}
    <py> swe_tool.py build <extract.jsonl> <instance_id> [...]       builds images (network, trusted)
"""
import json
import sys


def _instance(path, iid):
    for line in open(path):
        r = json.loads(line)
        if r["instance_id"] == iid:
            for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                if isinstance(r[k], str):
                    r[k] = json.loads(r[k])
            for k in ("problem_statement", "hints_text", "created_at"):
                r.setdefault(k, "")
            return r
    raise KeyError(iid)


def main(argv):
    from swebench.harness.test_spec.test_spec import make_test_spec
    cmd, path, iid = argv[0], argv[1], argv[2]
    inst = _instance(path, iid)
    ts = make_test_spec(inst, namespace=None)
    if cmd == "spec":
        print(json.dumps({"eval_script": ts.eval_script, "image_key": ts.instance_image_key,
                          "arch": ts.arch}))
    elif cmd == "grade":
        from swebench.harness.grading import get_eval_report
        patch = open(argv[4]).read()
        rep = get_eval_report(ts, {"instance_id": iid, "model_patch": patch,
                                   "model_name_or_path": "audit"}, argv[3], True)
        print(json.dumps({"resolved": bool(rep[iid]["resolved"]), "report": rep[iid]}))
    elif cmd == "build":
        import docker
        from swebench.harness.docker_build import build_instance_images
        ds = [_instance(path, i) for i in argv[2:]]
        build_instance_images(docker.from_env(), ds, force_rebuild=False, max_workers=1,
                              namespace=None, tag="latest")
    else:
        raise SystemExit(cmd)


if __name__ == "__main__":
    main(sys.argv[1:])
