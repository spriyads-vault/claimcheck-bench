"""Verified licences for every corpus this harness will ingest.

Nothing is ingested on a guess. Each entry here was checked against its primary
source before any bytes were mapped into the schema, and the check is recorded
with the date and the URL it was read from so a reader can re-run it rather than
take this file's word for it.

An entry with ``permits_this_use=False`` is a refusal, not a warning:
:func:`require_licence` raises and the ingest stops. A corpus with no entry at
all is also a refusal -- silence is not permission.
"""

from __future__ import annotations

from ..schemas import SourceLicence


class LicenceError(RuntimeError):
    """Raised when a corpus has no verified licence, or one that forbids this use."""


#: Checked 2026-09-20 against each project's own LICENSE file via the GitHub
#: licence API, and against the licence text the AppWorld packer writes into
#: every protected bundle.
LICENCES: dict[str, SourceLicence] = {
    "appworld": SourceLicence(
        spdx="Apache-2.0",
        name="Apache License 2.0",
        url="https://github.com/StonyBrookNLP/appworld/blob/main/LICENSE",
        permits_this_use=True,
        conditions=(
            "Apache-2.0: retain the licence and attribution notice.",
            # This is AppWorld's own wording, written into every bundle it packs.
            "The experiment-output bundle is the protected portion of AppWorld, "
            "released under Apache 2.0 with the additional requirement that any "
            "public redistribution of it, or of its derivatives, must also be in "
            "an encrypted format. This harness publishes nothing and the ingested "
            "corpus is git-ignored, so the condition is met by not redistributing.",
        ),
        attribution=(
            "AppWorld: A Controllable World of Apps and People for Benchmarking "
            "Interactive Coding Agents. Trivedi et al., ACL 2024 (Best Resource "
            "Paper). https://appworld.dev/ -- experiment outputs v0.1.3."
        ),
        verified_utc="2026-09-20",
        verified_from=(
            "https://api.github.com/repos/StonyBrookNLP/appworld/license "
            "(spdx_id Apache-2.0) and the LICENSE text packed into the bundle by "
            "scripts/release_experiment_outputs.py."
        ),
    ),
    "tau2": SourceLicence(
        spdx="MIT",
        name="MIT License",
        url="https://github.com/sierra-research/tau2-bench/blob/main/LICENSE",
        permits_this_use=True,
        conditions=("MIT: retain the copyright notice and permission notice.",),
        attribution=("tau2-bench, Sierra Research. https://github.com/sierra-research/tau2-bench"),
        verified_utc="2026-09-20",
        verified_from=(
            "https://api.github.com/repos/sierra-research/tau2-bench/license (spdx_id MIT)."
        ),
    ),
    # Recorded as a refusal on purpose. The dataset carries no licence tag and
    # no licence statement in its card, and it is access-gated. Unclear is not
    # permission, so this harness will not ingest it.
    "cx-cmu/agent_trajectories": SourceLicence(
        spdx="NOASSERTION",
        name="No licence declared",
        url="https://huggingface.co/datasets/cx-cmu/agent_trajectories",
        permits_this_use=False,
        conditions=(
            "The dataset card declares no licence: the HuggingFace API returns no "
            "licence tag and cardData carries only a pretty_name. The repository is "
            "additionally gated ('gated: auto'), so its terms cannot be read without "
            "accepting them. Unclear is not permission.",
        ),
        verified_utc="2026-09-20",
        verified_from=(
            "https://huggingface.co/api/datasets/cx-cmu/agent_trajectories?full=true "
            "-- no license tag, cardData {pretty_name}, gated=auto."
        ),
    ),
}


def require_licence(corpus: str) -> SourceLicence:
    """Return the verified licence for ``corpus``, or refuse.

    Called before a single byte of a corpus is read. A corpus that is not listed
    here has not been checked, and an unchecked corpus is not ingested.
    """
    licence = LICENCES.get(corpus)
    if licence is None:
        raise LicenceError(
            f"no verified licence on record for corpus {corpus!r}. Nothing is "
            "ingested on a guess: verify the licence against its primary source, "
            "add it to ingest/licences.py with the date and URL it was read from, "
            "and run this again."
        )
    if not licence.permits_this_use:
        raise LicenceError(
            f"the licence on record for {corpus!r} does not permit this use.\n"
            f"  licence  {licence.name} ({licence.spdx})\n"
            f"  source   {licence.url}\n"
            f"  checked  {licence.verified_utc} from {licence.verified_from}\n"
            + "".join(f"  reason   {c}\n" for c in licence.conditions)
            + "Refusing to ingest. Nothing was downloaded and nothing was written."
        )
    return licence
