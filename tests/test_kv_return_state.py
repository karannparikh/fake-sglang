import time
from dataclasses import replace

import pytest

from fakeengine.config import SimConfig
from fakeengine.kv_return import (
    KVReturnAbort,
    KVReturnCommit,
    KVReturnManager,
    KVReturnPrepare,
    KVReturnStatus,
    KVReturnTransferTicket,
    kv_return_checksum,
)
from fakeengine.kvcache import block_hashes
from fakeengine.scheduler import Scheduler


class RecordingEvents:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_stored(self, **kwargs):
        self.stored.append(kwargs)

    def publish_removed(self, block_ids):
        self.removed.extend(block_ids)


def _scheduler(
    server_args,
    *,
    rank=0,
    blocks=4,
    clock=None,
    ttl=300,
    max_terminal=1024,
    handle_timeout_ms=1000,
    transfer_deadline_ms=100,
):
    config = SimConfig.from_server_args(server_args(
        disaggregation_mode="prefill",
        enable_kv_return=True,
        fake_kv_return_max_tokens=8,
        fake_kv_return_handle_timeout_ms=handle_timeout_ms,
        fake_kv_return_deadline_ms=transfer_deadline_ms,
        fake_kv_return_bytes_per_token=16,
        fake_num_kv_blocks=blocks,
        page_size=4,
    ))
    events = RecordingEvents()
    scheduler = Scheduler(config, kv_events=events, dp_rank=rank)
    scheduler.kv_return = KVReturnManager(
        scheduler, rank, clock=clock or time.monotonic,
        terminal_ttl_s=ttl, max_terminal_records=max_terminal,
    )
    return scheduler, events


def _seed(scheduler, prompt=(1, 2, 3, 4), salt=None):
    hashes = block_hashes(prompt, scheduler.config.block_size, salt)
    blocks = scheduler.pool.allocate(len(hashes))
    assert blocks is not None
    scheduler.pool.insert_prefix(hashes, blocks)
    scheduler.pool.free(blocks)
    return hashes


def _ticket(scheduler, *, room=17, prompt=(1, 2, 3, 4), salt=None):
    result = scheduler.kv_return.issue_ticket(room, prompt, salt)
    assert result.status == KVReturnStatus.ISSUED
    assert result.ticket is not None
    return result.ticket


def _prepare(ticket, generated=(5, 6, 7, 8)):
    return KVReturnPrepare(
        ticket, ticket.attempt_id, ticket.compatibility.prefill_dp_rank,
        ticket.compatibility.fingerprint, ticket.cache_salt, len(generated),
        len(generated) * ticket.compatibility.kv_bytes_per_token,
        kv_return_checksum(ticket, generated),
    )


def _transfer_ticket(ticket, *, deadline_ms=100):
    return KVReturnTransferTicket(
        "transfer-nonce",
        ticket.nonce,
        ticket.attempt_id,
        deadline_ms,
    )


def _commit(ticket, transfer_ticket, generated=(5, 6, 7, 8)):
    return KVReturnCommit(
        _prepare(ticket, generated),
        transfer_ticket,
        tuple(generated),
    )


def _abort(ticket):
    return KVReturnAbort(ticket, ticket.attempt_id)


def _mismatched_prepare(prepare, case):
    compatibility_changes = {
        "model": {"served_model_name": "other-model"},
        "layout": {"kv_layout_fingerprint": "different-layout"},
        "block-size": {"block_size": 8},
        "bytes-per-token": {"kv_bytes_per_token": 32},
        "dp-rank": {"prefill_dp_rank": 1},
    }
    if case in compatibility_changes:
        contract = replace(
            prepare.ticket.compatibility, **compatibility_changes[case],
        )
        ticket = replace(prepare.ticket, compatibility=contract)
        return replace(
            prepare, ticket=ticket, prefill_dp_rank=contract.prefill_dp_rank,
            compatibility_fingerprint=contract.fingerprint,
        )
    changes = {
        "cache-salt": {"cache_salt": "tenant-b"},
        "attempt": {"attempt_id": "other-attempt"},
        "payload-bytes": {"claimed_payload_bytes": 63},
        "token-count": {
            "generated_token_count": 9, "claimed_payload_bytes": 144,
        },
    }
    return replace(prepare, **changes[case])


def test_valid_lifecycle_binds_contract_and_publishes_events(server_args):
    scheduler, events = _scheduler(server_args, rank=2)
    prompt = (1, 2, 3, 4)
    generated = (5, 6, 7, 8, 9, 10, 11, 12)
    _seed(scheduler, prompt, "tenant-a")
    ticket = _ticket(scheduler, prompt=prompt, salt="tenant-a")
    contract = ticket.compatibility
    assert (
        contract.protocol_version, contract.prefill_dp_rank,
        contract.served_model_name, contract.model_identity,
        contract.vocab_size, contract.block_size, contract.kv_bytes_per_token,
    ) == (
        scheduler.config.kv_return_protocol_version, 2,
        scheduler.config.served_model_name, scheduler.config.model_path,
        scheduler.config.vocab_size, scheduler.config.block_size, 16,
    )
    assert contract.kv_layout_fingerprint and contract.fingerprint
    assert ticket.max_payload_bytes == ticket.max_generated_tokens * 16
    assert ticket.handle_timeout_ms == 1000

    prepare = _prepare(ticket, generated)
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.transfer_ticket is not None
    committed = scheduler.kv_return.commit(
        _commit(ticket, prepared.transfer_ticket, generated)
    )
    hashes = block_hashes(prompt + generated, 4, "tenant-a")
    assert (prepared.status, prepared.reserved_blocks) == (
        KVReturnStatus.PREPARED, 2,
    )
    assert committed.status == KVReturnStatus.COMMITTED
    assert scheduler.pool.match_prefix(hashes) == 3
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == [{
        "token_ids": prompt + generated,
        "digests": hashes,
        "new_indices": [1, 2],
        "cache_salt": "tenant-a",
    }]
    assert events.removed == []


def test_prepare_rejects_capacity_before_transfer(server_args):
    scheduler, events = _scheduler(server_args, blocks=2)
    prompt_hashes = _seed(scheduler)
    held = scheduler.pool.allocate(1)
    assert held is not None
    result = scheduler.kv_return.prepare(_prepare(_ticket(scheduler)))
    assert result.status == KVReturnStatus.CAPACITY_REJECTED
    assert scheduler.pool.match_prefix(prompt_hashes) == 1
    assert events.stored == events.removed == []
    scheduler.pool.free(held)
    assert scheduler.pool.can_allocate(2)


@pytest.mark.parametrize(
    "case",
    [
        "model", "layout", "block-size", "bytes-per-token", "dp-rank",
        "cache-salt", "attempt", "payload-bytes", "token-count",
    ],
)
def test_prepare_rejects_mismatched_contract_and_controls(server_args, case):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler, salt="tenant-a")
    prepare = _prepare(_ticket(scheduler, salt="tenant-a"))
    result = scheduler.kv_return.prepare(_mismatched_prepare(prepare, case))
    assert result.status == KVReturnStatus.INCOMPATIBLE
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_expiry_releases_capacity_and_is_idempotent(server_args):
    now = [10.0]
    scheduler, _ = _scheduler(server_args, clock=lambda: now[0])
    _seed(scheduler)
    ticket = _ticket(scheduler)
    prepare = _prepare(ticket)
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.status == KVReturnStatus.PREPARED
    assert prepared.transfer_ticket is not None
    commit = _commit(ticket, prepared.transfer_ticket)
    now[0] += prepared.transfer_ticket.deadline_ms / 1000
    assert scheduler.kv_return.expire() == 1
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    retry = scheduler.kv_return.prepare(prepare)
    assert retry.status == KVReturnStatus.EXPIRED and retry.idempotent
    assert scheduler.kv_return.commit(commit).status == KVReturnStatus.EXPIRED


@pytest.mark.parametrize(
    "case,expected",
    [
        ("negative", KVReturnStatus.INVALID_TOKEN_IDS),
        ("boolean", KVReturnStatus.INVALID_TOKEN_IDS),
        ("out-of-vocab", KVReturnStatus.INVALID_TOKEN_IDS),
        ("control", KVReturnStatus.INCOMPATIBLE),
        ("checksum", KVReturnStatus.CHECKSUM_MISMATCH),
    ],
)
def test_commit_revalidates_tokens_checksum_and_controls(
    server_args, case, expected,
):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    ticket = _ticket(scheduler)
    generated = {
        "negative": (5, 6, 7, -1),
        "boolean": (5, 6, 7, True),
        "out-of-vocab": (5, 6, 7, 10**9),
    }.get(case, (5, 6, 7, 8))
    prepare = _prepare(ticket, generated)
    if case == "checksum":
        prepare = replace(prepare, token_ids_checksum="0" * 32)
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.status == KVReturnStatus.PREPARED
    assert prepared.transfer_ticket is not None
    commit_prepare = (
        replace(prepare, prefill_dp_rank=1) if case == "control" else prepare
    )
    result = scheduler.kv_return.commit(
        KVReturnCommit(
            commit_prepare,
            prepared.transfer_ticket,
            generated,
        )
    )
    assert result.status == expected
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_commit_rejects_payload_shorter_than_prepared(server_args):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    ticket = _ticket(scheduler)
    actual = (5, 6)
    prepare = replace(
        _prepare(ticket),
        token_ids_checksum=kv_return_checksum(ticket, actual),
    )
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.status == KVReturnStatus.PREPARED
    assert prepared.transfer_ticket is not None

    result = scheduler.kv_return.commit(
        KVReturnCommit(prepare, prepared.transfer_ticket, actual)
    )

    assert result.status == KVReturnStatus.INCOMPATIBLE
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_commit_rejects_a_different_transfer_ticket(server_args):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    ticket = _ticket(scheduler)
    prepare = _prepare(ticket)
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.transfer_ticket is not None
    wrong_ticket = replace(
        prepared.transfer_ticket,
        nonce="different-transfer-ticket",
    )

    result = scheduler.kv_return.commit(
        KVReturnCommit(
            prepare,
            wrong_ticket,
            (5, 6, 7, 8),
        )
    )

    assert result.status == KVReturnStatus.INCOMPATIBLE
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_commit_before_prepare_and_unissued_nonce_are_typed(server_args):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    ticket = _ticket(scheduler)
    commit = _commit(ticket, _transfer_ticket(ticket))
    first = scheduler.kv_return.commit(commit)
    second = scheduler.kv_return.commit(commit)
    assert first.status == second.status == KVReturnStatus.NOT_PREPARED
    assert not first.idempotent and second.idempotent
    unissued = replace(ticket, nonce="u" * 32)
    assert scheduler.kv_return.prepare(_prepare(unissued)).status == (
        KVReturnStatus.NOT_FOUND
    )
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_issued_only_expiry_is_stable_for_all_authenticated_calls(server_args):
    now = [10.0]
    scheduler, events = _scheduler(server_args, clock=lambda: now[0])
    _seed(scheduler)
    ticket = _ticket(scheduler)
    prepare = _prepare(ticket)
    now[0] += ticket.handle_timeout_ms / 1000
    assert scheduler.kv_return.expire() == 1

    commit = scheduler.kv_return.commit(
        KVReturnCommit(
            prepare,
            _transfer_ticket(ticket),
            (5, 6, 7, 8),
        )
    )
    retry_prepare = scheduler.kv_return.prepare(prepare)
    abort = scheduler.kv_return.abort(_abort(ticket))

    assert commit.status == retry_prepare.status == abort.status == (
        KVReturnStatus.EXPIRED
    )
    assert commit.idempotent and retry_prepare.idempotent and abort.idempotent
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []


def test_duplicate_prepare_commit_and_abort_are_stable(server_args):
    scheduler, events = _scheduler(server_args)
    _seed(scheduler)
    ticket = _ticket(scheduler)
    prepare = _prepare(ticket)
    first_prepare = scheduler.kv_return.prepare(prepare)
    second_prepare = scheduler.kv_return.prepare(prepare)
    assert first_prepare.transfer_ticket is not None
    commit = _commit(ticket, first_prepare.transfer_ticket)
    first_commit = scheduler.kv_return.commit(commit)
    second_commit = scheduler.kv_return.commit(commit)
    assert first_prepare.status == second_prepare.status == KVReturnStatus.PREPARED
    assert not first_prepare.idempotent and second_prepare.idempotent
    assert first_commit.status == second_commit.status == KVReturnStatus.COMMITTED
    assert not first_commit.idempotent and second_commit.idempotent
    assert len(events.stored) == 1

    abort_ticket = _ticket(scheduler, room=18)
    scheduler.kv_return.prepare(_prepare(abort_ticket))
    first_abort = scheduler.kv_return.abort(_abort(abort_ticket))
    second_abort = scheduler.kv_return.abort(_abort(abort_ticket))
    assert first_abort.status == second_abort.status == KVReturnStatus.ABORTED
    assert not first_abort.idempotent and second_abort.idempotent
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


def test_prepare_mints_fresh_deadline_after_long_generation(server_args):
    now = [10.0]
    scheduler, _ = _scheduler(
        server_args,
        clock=lambda: now[0],
        handle_timeout_ms=1000,
        transfer_deadline_ms=100,
    )
    _seed(scheduler)
    ticket = _ticket(scheduler)

    now[0] += 0.5
    prepared = scheduler.kv_return.prepare(_prepare(ticket))

    assert prepared.status == KVReturnStatus.PREPARED
    assert prepared.transfer_ticket is not None
    assert prepared.transfer_ticket.handle_nonce == ticket.nonce
    assert prepared.transfer_ticket.attempt_id == ticket.attempt_id
    assert prepared.transfer_ticket.deadline_ms == 100
    now[0] += 0.09
    assert scheduler.kv_return.commit(
        _commit(ticket, prepared.transfer_ticket)
    ).status == KVReturnStatus.COMMITTED


def test_terminal_records_have_ttl_and_hard_bound(server_args):
    now = [20.0]
    scheduler, _ = _scheduler(
        server_args, clock=lambda: now[0], ttl=5, max_terminal=2,
    )
    tickets = [_ticket(scheduler, room=room) for room in range(3)]
    for ticket in tickets:
        assert scheduler.kv_return.abort(_abort(ticket)).status == (
            KVReturnStatus.ABORTED
        )
    assert len(scheduler.kv_return._terminal) == 2
    assert scheduler.kv_return.abort(_abort(tickets[0])).status == (
        KVReturnStatus.NOT_FOUND
    )
    now[0] += 5
    scheduler.kv_return.expire()
    assert len(scheduler.kv_return._terminal) == 0


async def test_scheduler_shutdown_clears_all_manager_state(server_args):
    scheduler, _ = _scheduler(server_args)
    _seed(scheduler)
    prepared, issued, terminal = (
        _ticket(scheduler, room=room) for room in range(3)
    )
    scheduler.kv_return.prepare(_prepare(prepared))
    scheduler.kv_return.abort(_abort(terminal))
    assert issued.nonce
    await scheduler.stop()
    assert len(scheduler.kv_return._attempts) == len(scheduler.kv_return._terminal) == 0
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)


def test_atomic_import_failure_rolls_back_without_events(server_args):
    scheduler, events = _scheduler(server_args, blocks=3)
    prompt = (1, 2, 3, 4)
    prompt_hashes = _seed(scheduler, prompt)
    ticket = _ticket(scheduler, prompt=prompt)
    prepare = _prepare(ticket)
    prepared = scheduler.kv_return.prepare(prepare)
    assert prepared.transfer_ticket is not None
    commit = _commit(ticket, prepared.transfer_ticket)
    reservation = scheduler.kv_return._attempts[ticket.nonce].reservation
    conflicting = block_hashes((90, 91, 92, 93), 4)
    scheduler.pool.insert_prefix(
        conflicting, [reservation._reserved_blocks[0]],
    )
    result = scheduler.kv_return.commit(commit)
    returned = block_hashes(prompt + commit.generated_token_ids, 4)
    assert result.status == KVReturnStatus.IMPORT_FAILED
    assert scheduler.pool.match_prefix(returned) == len(prompt_hashes)
    assert scheduler.pool.match_prefix(conflicting) == 1
    assert scheduler.pool.can_allocate(scheduler.pool.num_blocks)
    assert events.stored == events.removed == []
