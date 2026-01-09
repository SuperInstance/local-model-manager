#!/usr/bin/env python3
"""
Test runner script for local-model-manager.

Provides convenient ways to run tests with different configurations:
- Run all tests
- Run specific test categories
- Run with coverage
- Run specific test files
- Generate coverage reports
"""

import subprocess
import sys
import argparse
from pathlib import Path


def run_command(cmd, description):
    """Run a command and print results."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\n❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"\n✅ {description} successful")
        return True


def run_all_tests(coverage=True, verbose=False, extra_args=None):
    """Run all tests with optional coverage."""
    cmd = ["python", "-m", "pytest"]

    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")  # Quiet mode

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-report=xml",
            "--cov-fail-under=80"
        ])

    if extra_args:
        cmd.extend(extra_args)

    cmd.append("tests/")

    return run_command(cmd, "All tests" + (" with coverage" if coverage else ""))


def run_unit_tests(coverage=True):
    """Run only unit tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "unit",
        "-v",
        "tests/"
    ]

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=term-missing"
        ])

    return run_command(cmd, "Unit tests")


def run_integration_tests(coverage=True):
    """Run only integration tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "integration",
        "-v",
        "tests/"
    ]

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=term-missing"
        ])

    return run_command(cmd, "Integration tests")


def run_slow_tests():
    """Run only slow tests."""
    cmd = [
        "python", "-m", "pytest",
        "-m", "slow",
        "-v",
        "-s",  # Show output
        "tests/"
    ]

    return run_command(cmd, "Slow tests")


def run_specific_test_file(test_file, coverage=False):
    """Run a specific test file."""
    test_path = Path(test_file)

    if not test_path.exists():
        # Try to find in tests directory
        test_path = Path(__file__).parent / test_file
        if not test_path.exists():
            print(f"❌ Test file not found: {test_file}")
            return False

    cmd = ["python", "-m", "pytest", "-v", str(test_path)]

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=term-missing"
        ])

    return run_command(cmd, f"Test file: {test_path.name}")


def run_specific_test(test_identifier, coverage=False):
    """Run a specific test."""
    cmd = ["python", "-m", "pytest", "-v", test_identifier]

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=term-missing"
        ])

    return run_command(cmd, f"Specific test: {test_identifier}")


def run_tests_by_pattern(pattern, coverage=False):
    """Run tests matching a pattern."""
    cmd = ["python", "-m", "pytest", "-v", "-k", pattern, "tests/"]

    if coverage:
        cmd.extend([
            "--cov=local_model_manager",
            "--cov-report=term-missing"
        ])

    return run_command(cmd, f"Tests matching pattern: {pattern}")


def generate_coverage_report():
    """Generate detailed coverage report."""
    print("\n📊 Generating coverage report...")

    cmd = [
        "python", "-m", "pytest",
        "--cov=local_model_manager",
        "--cov-report=html",
        "--cov-report=xml",
        "tests/"
    ]

    success = run_command(cmd, "Coverage report generation")

    if success:
        html_report = Path(__file__).parent.parent / "htmlcov" / "index.html"
        if html_report.exists():
            print(f"\n📈 HTML coverage report: {html_report}")
            print(f"   Open in browser: file://{html_report.absolute()}")

    return success


def run_with_xdist():
    """Run tests in parallel using pytest-xdist."""
    cmd = [
        "python", "-m", "pytest",
        "-n", "auto",  # Use all CPUs
        "-v",
        "tests/"
    ]

    return run_command(cmd, "Parallel tests (pytest-xdist)")


def list_tests():
    """List all available tests without running them."""
    cmd = [
        "python", "-m", "pytest",
        "--collect-only",
        "-q",
        "tests/"
    ]

    return run_command(cmd, "List all tests")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test runner for local-model-manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all tests with coverage
  python run_tests.py

  # Run all tests without coverage
  python run_tests.py --no-coverage

  # Run only unit tests
  python run_tests.py --unit

  # Run only integration tests
  python run_tests.py --integration

  # Run specific test file
  python run_tests.py --file test_model_loader.py

  # Run specific test
  python run_tests.py --test test_model_loader.py::TestGPUMemoryMonitor::test_init

  # Run tests matching pattern
  python run_tests.py --pattern "gpu"

  # Run only slow tests
  python run_tests.py --slow

  # List all tests
  python run_tests.py --list

  # Generate coverage report only
  python run_tests.py --coverage-report
        """
    )

    parser.add_argument("--all", action="store_true", help="Run all tests (default)")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--slow", action="store_true", help="Run only slow tests")
    parser.add_argument("--file", type=str, help="Run specific test file")
    parser.add_argument("--test", type=str, help="Run specific test")
    parser.add_argument("--pattern", type=str, help="Run tests matching pattern")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--coverage-report", action="store_true", help="Generate coverage report only")
    parser.add_argument("--parallel", action="store_true", help="Run tests in parallel (requires pytest-xdist)")
    parser.add_argument("--list", action="store_true", help="List all tests without running")
    parser.add_argument("--extra", nargs="*", help="Extra arguments to pass to pytest")

    args = parser.parse_args()

    # Check if pytest is installed
    try:
        import pytest
    except ImportError:
        print("❌ pytest is not installed. Please install it with:")
        print("   pip install pytest pytest-asyncio pytest-cov")
        sys.exit(1)

    # Determine which tests to run
    success = True

    if args.list:
        success = list_tests()
    elif args.coverage_report:
        success = generate_coverage_report()
    elif args.file:
        success = run_specific_test_file(args.file, coverage=not args.no_coverage)
    elif args.test:
        success = run_specific_test(args.test, coverage=not args.no_coverage)
    elif args.pattern:
        success = run_tests_by_pattern(args.pattern, coverage=not args.no_coverage)
    elif args.unit:
        success = run_unit_tests(coverage=not args.no_coverage)
    elif args.integration:
        success = run_integration_tests(coverage=not args.no_coverage)
    elif args.slow:
        success = run_slow_tests()
    elif args.parallel:
        success = run_with_xdist()
    else:
        # Default: run all tests
        success = run_all_tests(
            coverage=not args.no_coverage,
            verbose=args.verbose,
            extra_args=args.extra
        )

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
