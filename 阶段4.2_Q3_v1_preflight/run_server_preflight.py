"""Server qualification only: unit tests and real train/valid CPU smoke, no training."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import torch

from q3v1.scope_gate import EXPECTED_GATES, sha256, verify_scope_contract, verify_training_release


ROOT = Path(__file__).resolve().parent
SCOPE = ROOT / "stage_d_explanation_scope_finalization" / "explanation_scope_contract.json"


def run_logged(argv: list[str], out: Path, name: str, timeout: int) -> int:
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        result = subprocess.run(argv, cwd=ROOT, env=env, text=True, capture_output=True,
                                timeout=timeout, check=False)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout.decode(errors="replace") if isinstance(error.stdout, bytes) else (error.stdout or "")
        stderr = error.stderr.decode(errors="replace") if isinstance(error.stderr, bytes) else (error.stderr or "")
        stderr += f"\nTIMEOUT after {timeout} seconds\n"
        code = 124
    (out / f"{name}.stdout.log").write_text(stdout, encoding="utf-8")
    (out / f"{name}.stderr.log").write_text(stderr, encoding="utf-8")
    (out / f"{name}.exit_code.txt").write_text(f"{code}\n", encoding="utf-8")
    return code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aligned-pkl", type=Path, required=True)
    parser.add_argument("--expected-hostname", required=True,
                        help="hostname recorded on the actual execution server")
    parser.add_argument("--out", type=Path, default=ROOT / "reports" / "server_preflight")
    args = parser.parse_args()
    out = args.out.resolve()
    if not out.is_relative_to(ROOT):
        raise ValueError("preflight output must stay inside this Q3 package")
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("preflight output directory must be empty")
    out.mkdir(parents=True, exist_ok=True)
    verify_scope_contract(SCOPE)
    if platform.node() != args.expected_hostname:
        raise PermissionError("current host differs from the declared execution server")
    device = {"cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
              "python": platform.python_version(), "python_executable": sys.executable,
              "hostname": platform.node(), "platform": platform.platform()}
    if torch.cuda.is_available():
        device["cuda_devices"] = [{"index": i, "name": torch.cuda.get_device_name(i),
                                    "total_memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                                   for i in range(torch.cuda.device_count())]
    (out / "environment.json").write_text(json.dumps(device, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tests_code = run_logged([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"],
                            out, "unit_tests", 300)
    if tests_code != 0:
        raise RuntimeError("server unit tests failed; training gate remains CLOSED")
    smoke_code = run_logged([sys.executable, str(ROOT / "run_preflight.py"), "--aligned-pkl",
                             str(args.aligned_pkl), "--out", str(out)], out, "data_smoke", 1800)
    if smoke_code != 0:
        raise RuntimeError("server data/numeric smoke failed; training gate remains CLOSED")
    machine_path = out / "preflight_machine_report.json"
    machine = json.loads(machine_path.read_text(encoding="utf-8"))
    if machine.get("environment", {}).get("hostname") != device["hostname"]:
        raise RuntimeError("preflight report hostname differs from server environment")
    receipt = {
        "gate_version": "q3-server-preflight-v1",
        "status": "PASS",
        "gates": {**EXPECTED_GATES, "data": "PASS", "interface": "PASS", "xai": "PASS", "environment": "PASS"},
        "scope_contract_path": SCOPE.relative_to(ROOT).as_posix(),
        "scope_contract_sha256": sha256(SCOPE),
        "machine_report_path": machine_path.relative_to(ROOT).as_posix(),
        "machine_report_sha256": sha256(machine_path),
        "environment_report_path": (out / "environment.json").relative_to(ROOT).as_posix(),
        "environment_report_sha256": sha256(out / "environment.json"),
        "unit_tests_exit_code": tests_code,
        "unit_tests_exit_file_path": (out / "unit_tests.exit_code.txt").relative_to(ROOT).as_posix(),
        "unit_tests_exit_file_sha256": sha256(out / "unit_tests.exit_code.txt"),
        "data_smoke_exit_file_path": (out / "data_smoke.exit_code.txt").relative_to(ROOT).as_posix(),
        "data_smoke_exit_file_sha256": sha256(out / "data_smoke.exit_code.txt"),
        "test_used_for_model_selection": False,
        "attachment4_model_inference_run": False,
        "formal_training_run": False,
    }
    gate_path = out / "q3_server_preflight_gate.json"
    gate_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    verify_training_release(gate_path)
    print(json.dumps({"server_preflight_gate": "PASS", "gate_path": str(gate_path),
                      "cuda_available": device["cuda_available"], "formal_training_run": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()
