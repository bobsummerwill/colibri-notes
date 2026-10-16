"""Tiny dependency-free test runner shared by the hidden tests.

The grader copies this file, the test file and the model's code (as solution.py)
into a fresh temp dir and runs `python3 -I <test file>`. Output: one JSON line.
"""
import json
import signal
import sys
import traceback


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def run(tests, per_test_s=20):
    signal.signal(signal.SIGALRM, _alarm)
    results = []
    for i, (name, fn) in enumerate(tests):
        if i:  # cumulative progress line: if the outer timeout kills us, the grader keeps these counts
            print(json.dumps({"passed": sum(r["ok"] for r in results), "total": len(tests),
                              "partial": True, "results": results}), flush=True)
        signal.alarm(per_test_s)
        try:
            fn()
            results.append({"test": name, "ok": True})
        except _Timeout:
            results.append({"test": name, "ok": False, "error": f"timeout >{per_test_s}s"})
        except BaseException as e:  # noqa: BLE001 - a test failure of any kind
            tb = traceback.format_exception_only(type(e), e)[-1].strip()
            results.append({"test": name, "ok": False, "error": tb[:300]})
        finally:
            signal.alarm(0)
    passed = sum(r["ok"] for r in results)
    print(json.dumps({"passed": passed, "total": len(results), "results": results}))


def raises(exc, fn, *args):
    try:
        fn(*args)
    except exc:
        return
    except BaseException as e:  # wrong exception type
        raise AssertionError(f"expected {exc.__name__}, got {type(e).__name__}: {e}")
    raise AssertionError(f"expected {exc.__name__}, nothing raised")
