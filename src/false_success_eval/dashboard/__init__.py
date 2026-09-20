"""Local dashboard: the with-Jev / without-Jev view of a run.

Nothing in here reaches the network and nothing in here reaches for a key. The
browser is served aggregate scoring figures only: no request body, no response
body, no header, no environment value ever crosses into the page.
"""

from __future__ import annotations

from .aggregate import FOOTER_NOTICE, DashboardData, RunRef, discover_dashboard_runs

__all__ = ["FOOTER_NOTICE", "DashboardData", "RunRef", "discover_dashboard_runs"]
