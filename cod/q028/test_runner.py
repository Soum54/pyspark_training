import hashlib
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime
import pytest

ROOT = Path(__file__).resolve().parent
LOG = ROOT / "test_report.log"
CONFIG = ROOT / "test_config.json"
TEST_FILE = ROOT / "test_configurable_functions.py"
PROTECTED = [
    ROOT / "conftest.py",
    ROOT / "test_config.json",
    ROOT / "test_configurable_functions.py",
    ROOT / "test_runner.py",
    ROOT / "specification.md",
]

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

def expected_cases():
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        funcs = list(cfg.get("functions", []))
        names = list(cfg.get("test_names", []))
        count = int(cfg.get("test_count", 6))
    except Exception:
        funcs, names, count = [], [], 6
    if not names:
        names = funcs[:]
    while len(names) < count:
        names.append(f"assessment_validation_{len(names)+1}")
    return names[:count]

def reset_log(title="PySpark Assessment"):
    LOG.write_text(
        f"=== {title} Test Run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n",
        encoding="utf-8"
    )

def write_framework_failures(reason):
    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        title = cfg.get("assessment_name", "PySpark Assessment")
    except Exception:
        title = "PySpark Assessment"
    reset_log(title)
    clean = str(reason).replace("\n", " ").strip()
    with open(LOG, "a", encoding="utf-8") as f:
        for i, name in enumerate(expected_cases(), 1):
            f.write(f"[FAIL] Visible Test Case {i} Failed: {name}\n")
            f.write("Expected : Assessment function can be imported and evaluated\n")
            f.write(f"Actual   : {clean}\n")
            f.write("Reason   : Assessment execution could not evaluate this test case because the candidate solution/framework failed before normal validation.\n")
            f.write("Fix      : Correct the reported syntax/import/framework issue and rerun validation.\n")

def parse_counts():
    if not LOG.exists():
        return 0, 0
    text = LOG.read_text(encoding="utf-8", errors="replace")
    passed = len(re.findall(r"^\[PASS\].*Passed:", text, flags=re.MULTILINE))
    failed = len(re.findall(r"^\[FAIL\].*Failed:", text, flags=re.MULTILINE))
    return passed, failed

def print_report():
    if not LOG.exists():
        return
    for raw in LOG.read_text(encoding="utf-8", errors="replace").splitlines():
        if raw.startswith("[PASS]"):
            print(f"\033[92m{raw}\033[0m")
        elif raw.startswith("[FAIL]"):
            print(f"\033[91m{raw}\033[0m")
        else:
            print(raw)

def main():
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    try:
        cfg = json.loads(CONFIG.read_text(encoding="utf-8"))
        title = cfg.get("assessment_name", "PySpark Assessment")
        expected_total = int(cfg.get("test_count", 6))
    except Exception as e:
        write_framework_failures(f"Unable to read test_config.json: {e}")
        print_report()
        print("Summary: 0 Passed, 6 Failed")
        return 0

    reset_log(title)
    before = {str(p): sha256(p) for p in PROTECTED if p.exists()}

    # Candidate-facing execution: logical test failures go to test_report.log.
    # Set LEARNLYTICA_STRICT_QA=1 only during private release QA.
    pytest_code = pytest.main([TEST_FILE.name, "-v", "-s", "--tb=short"])

    changed = [
        p.name for p in PROTECTED
        if p.exists() and str(p) in before and sha256(p) != before[str(p)]
    ]
    if changed:
        write_framework_failures(
            "Protected assessment files were modified during execution: " + ", ".join(changed)
        )

    passed, failed = parse_counts()
    total = passed + failed

    # Convert syntax/import/collection errors into parser-compatible FAIL entries.
    if total != expected_total:
        write_framework_failures(
            f"pytest completed with code {pytest_code}, but test_report.log contained {total} "
            f"result entries; expected {expected_total}. Review syntax/import/collection output."
        )
        passed, failed = parse_counts()

    print_report()
    print(f"Summary: {passed} Passed, {failed} Failed")
    print(f"ASSESSMENT_SUMMARY | Passed={passed} | Failed={failed} | Total={passed+failed}")

    # Learnlytica integration contract: a candidate FAIL is an assessment result,
    # not a runner execution error. Non-zero is reserved for invalid report creation.
    if not LOG.exists() or (passed + failed) != expected_total:
        return 2
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
