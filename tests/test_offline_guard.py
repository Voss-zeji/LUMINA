from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path


class OfflineGuardTests(unittest.TestCase):
    def test_child_network_guard_blocks_before_any_connection_or_dns(self):
        script = (
            "import socket; assert socket.socket.connect.__module__ == 'sitecustomize';\n"
            "for call in [lambda:socket.getaddrinfo('example.invalid',443),"
            "lambda:socket.socket().connect(('example.invalid',443))]:\n"
            " try: call()\n"
            " except OSError as exc: assert 'network is disabled' in str(exc)\n"
            " else: raise AssertionError('unexpected network access')\n"
        )
        environment = dict(os.environ, LUMINA_TEST_OFFLINE="1", PYTHONPATH=str(Path(__file__).parent))
        child = subprocess.run([sys.executable, "-c", script], env=environment, capture_output=True,
                               timeout=10)
        self.assertEqual(child.returncode, 0, child.stderr.decode())
