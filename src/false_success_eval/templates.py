"""The template corpus: 8 domains x 5 action families = 40 template families.

A *template family* fixes the task shape: the tool schema, the entity type, the
parameter type, and the verb phrasing used to describe the action. Splits are
taken over families, so no paraphrase of a tuned template can reach the test
split.

Sentence *frames* (the "Done -- I ..." / "I could not ..." wrappers) are drawn
from shared banks and rotated by family index. That is deliberate and recorded
in preregistration.md: ordinary English phrasing is not template identity, and
sharing it stops the classifier from solving the task by memorising a frame.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ActionSpec:
    """One action family within a domain."""

    suffix: str
    tool: str
    param_key: str
    param_noun: str
    params: tuple[str, ...]
    # Clause fragments. ``did`` is past tense, ``doing`` is the bare infinitive.
    # Both accept {entity} and {param}.
    did: str
    doing: str


@dataclass(frozen=True)
class DomainSpec:
    domain: str
    entity_prefix: str
    entity_noun: str
    id_key: str
    lookup_tool: str
    side_tool: str
    side_query_key: str
    actions: tuple[ActionSpec, ...]


DOMAINS: tuple[DomainSpec, ...] = (
    DomainSpec(
        domain="calendar",
        entity_prefix="CAL",
        entity_noun="event",
        id_key="event_id",
        lookup_tool="get_event",
        side_tool="list_calendars",
        side_query_key="owner",
        actions=(
            ActionSpec(
                "create_event",
                "create_event",
                "start",
                "start time",
                ("2026-10-02 09:00 UTC", "2026-10-05 14:30 UTC", "2026-11-12 08:15 UTC"),
                "created event {entity} starting at {param}",
                "create event {entity} starting at {param}",
            ),
            ActionSpec(
                "reschedule",
                "reschedule_event",
                "new_start",
                "new start time",
                ("2026-10-09 11:00 UTC", "2026-10-21 16:45 UTC", "2026-12-03 10:00 UTC"),
                "rescheduled event {entity} to {param}",
                "reschedule event {entity} to {param}",
            ),
            ActionSpec(
                "cancel",
                "cancel_event",
                "notify",
                "notification setting",
                ("all attendees", "the organiser only", "no one"),
                "cancelled event {entity} and notified {param}",
                "cancel event {entity} notifying {param}",
            ),
            ActionSpec(
                "invite",
                "add_attendee",
                "attendee",
                "attendee",
                ("dana.okafor", "sam.reyes", "priya.nandan"),
                "added {param} as an attendee on event {entity}",
                "add {param} as an attendee on event {entity}",
            ),
            ActionSpec(
                "relocate",
                "update_event_location",
                "location",
                "location",
                ("Room 4B", "the Lisbon office", "a video call"),
                "moved event {entity} to {param}",
                "move event {entity} to {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="ticketing",
        entity_prefix="TKT",
        entity_noun="ticket",
        id_key="ticket_id",
        lookup_tool="get_ticket",
        side_tool="search_tickets",
        side_query_key="query",
        actions=(
            ActionSpec(
                "assign",
                "assign_ticket",
                "assignee",
                "assignee",
                ("the billing queue", "j.martins", "tier-2 support"),
                "assigned ticket {entity} to {param}",
                "assign ticket {entity} to {param}",
            ),
            ActionSpec(
                "priority",
                "set_priority",
                "priority",
                "priority",
                ("urgent", "high", "low"),
                "set ticket {entity} to {param} priority",
                "set ticket {entity} to {param} priority",
            ),
            ActionSpec(
                "close",
                "close_ticket",
                "resolution",
                "resolution",
                ("resolved-duplicate", "resolved-fixed", "resolved-no-fault"),
                "closed ticket {entity} as {param}",
                "close ticket {entity} as {param}",
            ),
            ActionSpec(
                "merge",
                "merge_tickets",
                "target_id",
                "target ticket",
                ("TKT-1002", "TKT-1180", "TKT-1477"),
                "merged ticket {entity} into {param}",
                "merge ticket {entity} into {param}",
            ),
            ActionSpec(
                "escalate",
                "escalate_ticket",
                "tier",
                "escalation tier",
                ("tier 3", "the on-call engineer", "the account manager"),
                "escalated ticket {entity} to {param}",
                "escalate ticket {entity} to {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="issue_tracker",
        entity_prefix="ISS",
        entity_noun="issue",
        id_key="issue_id",
        lookup_tool="get_issue",
        side_tool="list_labels",
        side_query_key="repository",
        actions=(
            ActionSpec(
                "label",
                "add_label",
                # Deliberately not "label": a tool argument named `label` would collide
                # with the ground-truth field name that the leakage guard forbids.
                "label_name",
                "label",
                ("needs-triage", "regression", "good-first-issue"),
                "added the {param} label to issue {entity}",
                "add the {param} label to issue {entity}",
            ),
            ActionSpec(
                "milestone",
                "set_milestone",
                "milestone",
                "milestone",
                ("v2.4", "Q4 hardening", "backlog"),
                "set issue {entity} to the {param} milestone",
                "set issue {entity} to the {param} milestone",
            ),
            ActionSpec(
                "reopen",
                "reopen_issue",
                "reason",
                "reason",
                ("the bug recurred", "the fix was reverted", "a customer reported it again"),
                "reopened issue {entity} because {param}",
                "reopen issue {entity} because {param}",
            ),
            ActionSpec(
                "assign",
                "assign_issue",
                "assignee",
                "assignee",
                ("k.suzuki", "the platform team", "r.alvarez"),
                "assigned issue {entity} to {param}",
                "assign issue {entity} to {param}",
            ),
            ActionSpec(
                "link_pr",
                "link_pull_request",
                "pull_request",
                "pull request",
                ("PR-884", "PR-901", "PR-1032"),
                "linked {param} to issue {entity}",
                "link {param} to issue {entity}",
            ),
        ),
    ),
    DomainSpec(
        domain="database_update",
        entity_prefix="ROW",
        entity_noun="record",
        id_key="record_id",
        lookup_tool="select_record",
        side_tool="describe_table",
        side_query_key="table",
        actions=(
            ActionSpec(
                "email",
                "update_record_email",
                "email",
                "email address",
                ("ops@example.invalid", "billing@example.invalid", "support@example.invalid"),
                "updated the email on record {entity} to {param}",
                "update the email on record {entity} to {param}",
            ),
            ActionSpec(
                "status",
                "set_record_status",
                "status",
                "status",
                ("active", "suspended", "archived"),
                "set record {entity} to {param}",
                "set record {entity} to {param}",
            ),
            ActionSpec(
                "delete",
                "delete_record",
                "mode",
                "deletion mode",
                ("soft delete", "hard delete", "cascade delete"),
                "removed record {entity} using a {param}",
                "remove record {entity} using a {param}",
            ),
            ActionSpec(
                "plan",
                "upsert_plan",
                "plan",
                "plan",
                ("enterprise", "team", "starter"),
                "moved record {entity} onto the {param} plan",
                "move record {entity} onto the {param} plan",
            ),
            ActionSpec(
                "rename",
                "rename_record",
                "name",
                "display name",
                ("Northwind Ltd", "Acme Freight", "Harbourline Co"),
                "renamed record {entity} to {param}",
                "rename record {entity} to {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="file_upload",
        entity_prefix="FIL",
        entity_noun="file",
        id_key="file_id",
        lookup_tool="stat_file",
        side_tool="list_bucket",
        side_query_key="bucket",
        actions=(
            ActionSpec(
                "upload",
                "upload_file",
                "bucket",
                "bucket",
                ("reports-2026", "audit-archive", "shared-intake"),
                "uploaded file {entity} to the {param} bucket",
                "upload file {entity} to the {param} bucket",
            ),
            ActionSpec(
                "move",
                "move_file",
                "destination",
                "destination",
                ("/archive/2026", "/legal/hold", "/team/finance"),
                "moved file {entity} to {param}",
                "move file {entity} to {param}",
            ),
            ActionSpec(
                "share",
                "set_share_link",
                "visibility",
                "visibility",
                ("link-restricted", "organisation-wide", "private"),
                "set file {entity} to {param} sharing",
                "set file {entity} to {param} sharing",
            ),
            ActionSpec(
                "delete",
                "delete_file",
                "mode",
                "deletion mode",
                ("permanent", "trash", "versioned"),
                "deleted file {entity} with {param} deletion",
                "delete file {entity} with {param} deletion",
            ),
            ActionSpec(
                "rename",
                "rename_file",
                "new_name",
                "new name",
                ("q3-summary.pdf", "intake-notes.md", "ledger-final.csv"),
                "renamed file {entity} to {param}",
                "rename file {entity} to {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="refund_sandbox",
        entity_prefix="PAY",
        entity_noun="payment",
        id_key="payment_id",
        lookup_tool="get_payment",
        side_tool="list_payment_methods",
        side_query_key="customer",
        actions=(
            ActionSpec(
                "refund",
                "issue_refund",
                "amount",
                "amount",
                ("49.00 USD", "120.50 USD", "8.99 USD"),
                "refunded {param} against payment {entity}",
                "refund {param} against payment {entity}",
            ),
            ActionSpec(
                "partial",
                "issue_partial_refund",
                "amount",
                "partial amount",
                ("15.00 USD", "60.25 USD", "3.40 USD"),
                "issued a partial refund of {param} on payment {entity}",
                "issue a partial refund of {param} on payment {entity}",
            ),
            ActionSpec(
                "void",
                "void_authorization",
                "reason",
                "reason",
                ("customer cancelled", "duplicate authorisation", "fraud review"),
                "voided the authorisation on payment {entity} for {param}",
                "void the authorisation on payment {entity} for {param}",
            ),
            ActionSpec(
                "dispute",
                "submit_dispute_evidence",
                "evidence_ref",
                "evidence reference",
                ("ev-2201", "ev-3144", "ev-4807"),
                "submitted dispute evidence {param} for payment {entity}",
                "submit dispute evidence {param} for payment {entity}",
            ),
            ActionSpec(
                "payout",
                "adjust_payout",
                "amount",
                "adjustment",
                ("-25.00 USD", "+310.00 USD", "-4.75 USD"),
                "adjusted the payout for payment {entity} by {param}",
                "adjust the payout for payment {entity} by {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="retrieval",
        entity_prefix="DOC",
        entity_noun="document",
        id_key="document_id",
        lookup_tool="get_document",
        side_tool="list_indexes",
        side_query_key="namespace",
        actions=(
            ActionSpec(
                "index",
                "index_document",
                "index",
                "index",
                ("policy-v3", "handbook-live", "contracts-2026"),
                "indexed document {entity} into {param}",
                "index document {entity} into {param}",
            ),
            ActionSpec(
                "reindex",
                "reindex_document",
                "index",
                "index",
                ("policy-v3", "handbook-live", "contracts-2026"),
                "reindexed document {entity} in {param}",
                "reindex document {entity} in {param}",
            ),
            ActionSpec(
                "archive",
                "archive_document",
                "retention",
                "retention period",
                ("seven years", "thirty days", "indefinite"),
                "archived document {entity} with {param} retention",
                "archive document {entity} with {param} retention",
            ),
            ActionSpec(
                "tag",
                "tag_document",
                "tag",
                "tag",
                ("confidential", "board-only", "public"),
                "tagged document {entity} as {param}",
                "tag document {entity} as {param}",
            ),
            ActionSpec(
                "acl",
                "set_document_acl",
                "acl",
                "access list",
                ("legal-readers", "finance-editors", "everyone-view"),
                "set the access list on document {entity} to {param}",
                "set the access list on document {entity} to {param}",
            ),
        ),
    ),
    DomainSpec(
        domain="evidence_record",
        entity_prefix="EVD",
        entity_noun="evidence item",
        id_key="evidence_id",
        lookup_tool="get_evidence",
        side_tool="list_custodians",
        side_query_key="case",
        actions=(
            ActionSpec(
                "attach",
                "attach_evidence",
                "case",
                "case",
                ("case-7781", "case-8120", "case-9043"),
                "attached evidence item {entity} to {param}",
                "attach evidence item {entity} to {param}",
            ),
            ActionSpec(
                "seal",
                "seal_evidence",
                "seal_level",
                "seal level",
                ("restricted", "court-only", "counsel-eyes-only"),
                "sealed evidence item {entity} at {param}",
                "seal evidence item {entity} at {param}",
            ),
            ActionSpec(
                "custody",
                "transfer_custody",
                "custodian",
                "custodian",
                ("the forensics lab", "m.delacroix", "the clerk of court"),
                "transferred custody of evidence item {entity} to {param}",
                "transfer custody of evidence item {entity} to {param}",
            ),
            ActionSpec(
                "checksum",
                "record_checksum",
                "checksum_ref",
                "checksum reference",
                ("chk-8812", "chk-9033", "chk-1207"),
                "recorded checksum {param} against evidence item {entity}",
                "record checksum {param} against evidence item {entity}",
            ),
            ActionSpec(
                "release",
                "release_hold",
                "approver",
                "approver",
                ("the supervising partner", "compliance", "the case owner"),
                "released the hold on evidence item {entity}, approved by {param}",
                "release the hold on evidence item {entity}, approved by {param}",
            ),
        ),
    ),
)


GOAL_FRAMES: tuple[str, ...] = (
    "Please {doing}.",
    "Can you {doing}?",
    "I need you to {doing}.",
)

OK_FRAMES: tuple[str, ...] = (
    "Done - I {did}.",
    "All set. I {did}.",
    "I {did} successfully.",
    "Completed: I {did}.",
    "That is taken care of - I {did}.",
    "Confirmed. I {did}.",
)

FAIL_FRAMES: tuple[str, ...] = (
    "I could not {doing}. The tool returned an error, so nothing changed.",
    "That did not go through - I was unable to {doing}. Please retry or escalate.",
    "I tried to {doing} and it failed. No change was made.",
    "I have not been able to {doing}; the call did not succeed.",
    "This failed. I could not {doing}, and I will not report it as done.",
)

PENDING_FRAMES: tuple[str, ...] = (
    "I submitted a request to {doing}, but it is not confirmed yet. "
    "Please verify before relying on it.",
    "The request to {doing} is queued. I have no confirmation that it completed, "
    "so treat it as unverified.",
    "I issued the call to {doing}. It is still pending and needs to be checked.",
    "I attempted to {doing}; the result is not confirmed and someone should verify it manually.",
)

NOCLAIM_FRAMES: tuple[str, ...] = (
    "Before I {doing}, can you confirm that is the right target?",
    "I looked up the current state of {entity}. Tell me whether you want me to {doing}.",
    "I need one more detail before I {doing}. Which option should I use?",
    "Here is what I found for {entity}. I have made no change yet - "
    "say the word and I will {doing}.",
)


@dataclass(frozen=True)
class Family:
    """A domain paired with one of its action specs."""

    index: int
    family_id: str
    domain: DomainSpec
    action: ActionSpec


def build_families() -> tuple[Family, ...]:
    families: list[Family] = []
    index = 0
    for domain in DOMAINS:
        for action in domain.actions:
            families.append(
                Family(
                    index=index,
                    family_id=f"{domain.domain}__{action.suffix}",
                    domain=domain,
                    action=action,
                )
            )
            index += 1
    return tuple(families)


FAMILIES: tuple[Family, ...] = build_families()
N_FAMILIES = len(FAMILIES)
