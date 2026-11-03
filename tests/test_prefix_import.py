from fakeengine.config import SimConfig
from fakeengine.kvcache import block_hashes, external_id
from fakeengine.scheduler import Scheduler


class RecordingEvents:
    def __init__(self):
        self.stored = []
        self.removed = []

    def publish_stored(self, **kwargs):
        self.stored.append(kwargs)

    def publish_removed(self, block_ids):
        self.removed.extend(block_ids)


def _scheduler(server_args, blocks=4):
    config = SimConfig.from_server_args(server_args(
        fake_num_kv_blocks=blocks,
        page_size=4,
    ))
    events = RecordingEvents()
    return Scheduler(config, kv_events=events), events


def _seed(scheduler, token_ids, salt=None):
    hashes = block_hashes(token_ids, scheduler.config.block_size, salt)
    blocks = scheduler.pool.allocate(len(hashes))
    assert blocks is not None
    scheduler.pool.insert_prefix(hashes, blocks)
    scheduler.pool.free(blocks)
    return hashes


def test_exact_capacity_reservation_and_release(server_args):
    scheduler, events = _scheduler(server_args, blocks=3)
    prompt = (1, 2, 3, 4)
    _seed(scheduler, prompt)

    reservation = scheduler.reserve_prefix_import(prompt, 8)

    assert reservation is not None
    assert reservation.reserved_blocks == 2
    assert scheduler.pool.allocate(1) is None
    assert scheduler.release_prefix_import(reservation)
    assert not scheduler.release_prefix_import(reservation)
    assert scheduler.pool.can_allocate(3)
    assert events.stored == events.removed == []


def test_capacity_rejection_does_not_evict_or_publish(server_args):
    scheduler, events = _scheduler(server_args, blocks=2)
    prompt = (1, 2, 3, 4)
    prompt_hashes = _seed(scheduler, prompt)
    held = scheduler.pool.allocate(1)
    assert held is not None

    assert scheduler.reserve_prefix_import(prompt, 4) is None
    assert scheduler.pool.match_prefix(prompt_hashes) == 1
    assert events.stored == events.removed == []
    scheduler.pool.free(held)
    assert scheduler.pool.can_allocate(2)


def test_partial_block_boundary_only_imports_full_blocks(server_args):
    prompt = (1, 2, 3, 4, 5)

    scheduler, events = _scheduler(server_args)
    _seed(scheduler, prompt)
    reservation = scheduler.reserve_prefix_import(prompt, 2)
    assert reservation is not None
    assert reservation.reserved_blocks == 0
    assert scheduler.commit_prefix_import(reservation, (6, 7))
    assert events.stored == []

    reservation = scheduler.reserve_prefix_import(prompt, 3)
    assert reservation is not None
    assert reservation.reserved_blocks == 1
    assert scheduler.commit_prefix_import(reservation, (6, 7, 8))
    assert len(events.stored) == 1
    assert events.stored[0]["new_indices"] == [1]
    assert events.stored[0]["token_ids"][4:8] == (5, 6, 7, 8)


def test_cache_salt_is_part_of_prompt_identity(server_args):
    scheduler, events = _scheduler(server_args)
    prompt = (1, 2, 3, 4)
    _seed(scheduler, prompt, "tenant-a")

    assert scheduler.reserve_prefix_import(prompt, 4, "tenant-b") is None
    reservation = scheduler.reserve_prefix_import(prompt, 4, "tenant-a")
    assert reservation is not None
    assert scheduler.release_prefix_import(reservation)
    assert events.stored == events.removed == []


def test_failed_atomic_import_publishes_nothing_and_rolls_back(server_args):
    scheduler, events = _scheduler(server_args, blocks=3)
    prompt = (1, 2, 3, 4)
    prompt_hashes = _seed(scheduler, prompt)
    reservation = scheduler.reserve_prefix_import(prompt, 4)
    assert reservation is not None

    conflicting = block_hashes((90, 91, 92, 93), 4)
    scheduler.pool.insert_prefix(
        conflicting, [reservation._reserved_blocks[0]],
    )

    assert not scheduler.commit_prefix_import(reservation, (5, 6, 7, 8))
    returned = block_hashes(prompt + (5, 6, 7, 8), 4)
    assert scheduler.pool.match_prefix(returned) == len(prompt_hashes)
    assert scheduler.pool.match_prefix(conflicting) == 1
    assert events.stored == events.removed == []
    assert scheduler.pool.can_allocate(3)


def test_invalid_commit_releases_reserved_capacity(server_args):
    scheduler, events = _scheduler(server_args, blocks=3)
    prompt = (1, 2, 3, 4)
    _seed(scheduler, prompt)
    reservation = scheduler.reserve_prefix_import(prompt, 4)
    assert reservation is not None

    assert not scheduler.commit_prefix_import(
        reservation, (5, 6, 7, 8, 9),
    )
    assert scheduler.pool.can_allocate(3)
    assert events.stored == events.removed == []


def test_already_resident_extension_blocks_are_not_republished(server_args):
    scheduler, events = _scheduler(server_args, blocks=5)
    prompt = (1, 2, 3, 4)
    generated = (5, 6, 7, 8, 9, 10, 11, 12)
    _seed(scheduler, prompt + generated[:4])

    reservation = scheduler.reserve_prefix_import(prompt, len(generated))
    assert reservation is not None
    assert scheduler.commit_prefix_import(reservation, generated)

    hashes = block_hashes(prompt + generated, 4)
    assert scheduler.pool.match_prefix(hashes) == 3
    assert len(events.stored) == 1
    assert events.stored[0]["new_indices"] == [2]
    assert events.removed == []
    assert scheduler.pool.can_allocate(5)


def test_import_events_remain_truthful_through_eviction(server_args):
    scheduler, events = _scheduler(server_args, blocks=2)
    prompt = (1, 2, 3, 4)
    generated = (5, 6, 7, 8)
    _seed(scheduler, prompt)
    reservation = scheduler.reserve_prefix_import(prompt, len(generated))
    assert reservation is not None

    assert scheduler.commit_prefix_import(reservation, generated)
    combined_hashes = block_hashes(prompt + generated, 4)
    assert len(events.stored) == 1
    assert events.stored[0]["new_indices"] == [1]

    block = scheduler.pool.allocate(1)
    assert block is not None
    assert events.removed == [external_id(combined_hashes[-1])]
    assert scheduler.pool.match_prefix(combined_hashes) == 1
