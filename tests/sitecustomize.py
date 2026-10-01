"""Opt-in offline test guard, inherited by the test runner's child interpreters."""
import os

if os.environ.get("LUMINA_TEST_OFFLINE") == "1":
    import socket

    def _blocked(*args, **kwargs):
        raise OSError("network is disabled during LUMINA mock tests")

    socket.socket.connect = _blocked
    socket.socket.connect_ex = _blocked
    socket.socket.sendto = _blocked
    socket.create_connection = _blocked
    socket.getaddrinfo = _blocked
