"""
ΔP-42 Sovereign Cognitive Core (Gate 1) - Turnkey Standalone E2E Test Runner
Executes the comprehensive 4-tier opaque-box test suite, tracks progressive milestone
readiness, and reports structured metrics with exit code discipline.
"""

from __future__ import annotations

import io
import os
import sys
import time
from typing import Dict, List, Tuple

# Ensure project root is in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest

# Ensure robust UTF-8 encoding across Windows console environments
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def print_banner() -> None:
    banner = """
================================================================================
   DELTA P-42 SOVEREIGN COGNITIVE CORE (GATE 1) -- END-TO-END RUNNER
================================================================================
  Target: Stage-Gate 1 (Cognitive Core & DVP-01 Quantitative Verification)
  Engine: Hybrid Relaxation Oscillator (FitzHugh-Nagumo Class) in SI Radians
  Scope:  4 Tiers | 29 Features in PROJECT.md | Zero-Downgrade Enforcement
================================================================================
"""
    print(banner)


class TierReportPlugin:
    """Pytest plugin that aggregates test outcomes by tier."""

    def __init__(self) -> None:
        self.passed: int = 0
        self.failed: int = 0
        self.skipped: int = 0
        self.tier_stats: Dict[str, Dict[str, int]] = {
            "Tier 1 (Feature Coverage)": {"passed": 0, "failed": 0, "skipped": 0},
            "Tier 2 (Boundaries & Corners)": {"passed": 0, "failed": 0, "skipped": 0},
            "Tier 3 (Cross Interactions)": {"passed": 0, "failed": 0, "skipped": 0},
            "Tier 4 (Real-World Scenarios)": {"passed": 0, "failed": 0, "skipped": 0},
        }

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call" and not (report.when == "setup" and report.skipped):
            return

        tier_name = "Unknown"
        nodeid = report.nodeid
        if "test_tier1" in nodeid:
            tier_name = "Tier 1 (Feature Coverage)"
        elif "test_tier2" in nodeid:
            tier_name = "Tier 2 (Boundaries & Corners)"
        elif "test_tier3" in nodeid:
            tier_name = "Tier 3 (Cross Interactions)"
        elif "test_tier4" in nodeid:
            tier_name = "Tier 4 (Real-World Scenarios)"

        if report.passed:
            self.passed += 1
            if tier_name in self.tier_stats:
                self.tier_stats[tier_name]["passed"] += 1
        elif report.failed:
            self.failed += 1
            if tier_name in self.tier_stats:
                self.tier_stats[tier_name]["failed"] += 1
        elif report.skipped:
            self.skipped += 1
            if tier_name in self.tier_stats:
                self.tier_stats[tier_name]["skipped"] += 1


def run_e2e_suite(args: List[str] = None) -> int:
    """Executes the E2E test suite and displays structured telemetry."""
    print_banner()

    test_dir = os.path.dirname(os.path.abspath(__file__))
    plugin = TierReportPlugin()

    pytest_args = [
        test_dir,
        "-q",
        "--tb=short",
    ]
    if args:
        pytest_args.extend(args)

    start_time = time.perf_counter()
    exit_code = pytest.main(pytest_args, plugins=[plugin])
    elapsed_time = time.perf_counter() - start_time

    # Display Tier Breakdown Table
    print("\n" + "=" * 80)
    print("                      E2E TIER BREAKDOWN & STATUS")
    print("=" * 80)
    print(f" {'Tier Name':<32} | {'Passed':<8} | {'Skipped':<8} | {'Failed':<8} | {'Status':<10}")
    print("-" * 80)

    for tier, counts in plugin.tier_stats.items():
        p = counts["passed"]
        s = counts["skipped"]
        f = counts["failed"]
        status = "FAIL" if f > 0 else ("PASS (M1)" if s > 0 else "PASS")
        print(f" {tier:<32} | {p:<8} | {s:<8} | {f:<8} | {status:<10}")

    print("-" * 80)
    total_tests = plugin.passed + plugin.skipped + plugin.failed
    overall_status = "100% OK" if (plugin.failed == 0 and total_tests > 0) else ("DEFECTS" if plugin.failed > 0 else "NO TESTS")
    print(f" {'TOTAL ACROSS ALL TIERS':<32} | {plugin.passed:<8} | {plugin.skipped:<8} | {plugin.failed:<8} | {overall_status}")
    print("=" * 80)
    print(f" Execution Duration: {elapsed_time:.3f} seconds")
    print(f" Progressive Testing Status: {'ALL ACTIVE MILESTONE TESTS PASSED' if plugin.failed == 0 else 'TEST FAILURES DETECTED'}")
    print("=" * 80 + "\n")

    return 0 if (plugin.failed == 0 and plugin.passed > 0) else 1


if __name__ == "__main__":
    sys.exit(run_e2e_suite(sys.argv[1:]))
