"""Backward-compatible import for the packaged downstream tester."""

if __package__:
    from .Hyperfree_test import HyperspectralTester
else:  # pragma: no cover - direct execution from the release directory
    from Hyperfree_test import HyperspectralTester

__all__ = ["HyperspectralTester"]
