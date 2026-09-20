"""Ingestion of real, labelled agent-trace corpora.

Three rules, in this order:

1. **Licence first.** :func:`~.licences.require_licence` runs before a byte is
   read. A corpus with no verified licence, or one whose licence does not permit
   this use, stops the ingest with an explanation. Nothing is guessed.
2. **Labels from ground truth, never from text.** See :mod:`.labelling`. The
   rule is exported as data so the same wording reaches the provenance file, the
   CLI and the dashboard caveat.
3. **Splits are task-disjoint.** Paraphrases of one task share a split unit, so
   a rewording of a training task cannot appear in test.
"""

from __future__ import annotations

from .licences import LICENCES, LicenceError, require_licence

__all__ = ["LICENCES", "LicenceError", "require_licence"]
