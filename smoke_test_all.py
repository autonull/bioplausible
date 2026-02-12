#!/usr/bin/env python3
"""
Comprehensive Smoke Test Runner
Runs bioplausible/tests/test_registry_smoke.py to verify all models against all compatible tasks.
"""

import sys
import unittest
from bioplausible.tests.test_registry_smoke import TestRegistrySmoke

if __name__ == "__main__":
    print("\nRunning Full Registry Smoke Tests...")
    print("This will test every model in MODEL_REGISTRY against every compatible task.")

    # Load tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestRegistrySmoke)

    # Run
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        sys.exit(1)

    print("\n✅ All smoke tests passed!")
