# SPDX-License-Identifier: MIT
"""Image pixels must never alias a token-only prefix-cache entry."""

import pytest
from conftest import MockConfig

from atom.model_engine.block_manager import BlockManager
from atom.model_engine.scheduler import ScheduledBatchOutput, Scheduler


def test_images_do_not_reuse_text_cache_but_text_still_does(seq_factory):
    bm = BlockManager(MockConfig(enable_prefix_caching=True, num_kvcache_blocks=100))
    tokens = list(range(12))
    text = seq_factory(tokens)
    bm.allocate(text)
    bm.hash_blocks(text, len(tokens))
    bm.deallocate(text)
    assert bm.can_allocate(seq_factory(tokens)) > 0
    for pixels in [b"red", b"blue"]:
        image = seq_factory(tokens, multimodal_data={"pixel_values": pixels})
        assert bm.can_allocate(image) == 0


def test_image_cache_is_not_published_after_batch_consumes_pixels(seq_factory):
    bm = BlockManager(MockConfig(enable_prefix_caching=True, num_kvcache_blocks=100))
    tokens = list(range(12))
    image = seq_factory(tokens, multimodal_data={"pixel_values": b"red"})
    bm.allocate(image)
    image.multimodal_data = None  # ScheduledBatch transfers the payload once.
    bm.hash_blocks(image, len(tokens))
    bm.hash_decode_blocks(image, len(tokens))
    bm.deallocate(image)
    assert bm.can_allocate(seq_factory(tokens)) == 0


def test_preempted_image_is_prefilled_with_its_pixels_again(seq_factory):
    scheduler = Scheduler(MockConfig(num_kvcache_blocks=100))
    pixels = {"pixel_values": object()}
    image = seq_factory(list(range(8)), multimodal_data=pixels)
    scheduler.add(image)
    first, _ = scheduler.schedule()
    assert first.multimodal_data[image.id] is pixels
    assert image.multimodal_data is None
    scheduler.running.remove(image)
    assert scheduler.preempt(image)
    second, _ = scheduler.schedule()
    assert second.multimodal_data[image.id] is pixels
    assert list(second.num_scheduled_tokens) == [8]


@pytest.mark.parametrize("advance_on_schedule", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_preempted_image_replays_generated_suffix_in_chunks(
    seq_factory, advance_on_schedule, chunked
):
    scheduler = Scheduler(
        MockConfig(
            num_kvcache_blocks=100,
            max_num_batched_tokens=8,
            enable_chunked_prefill=chunked,
        )
    )
    scheduler.advance_on_schedule = advance_on_schedule
    pixels = {"pixel_values": object()}
    image = seq_factory(list(range(8)), multimodal_data=pixels)
    scheduler.add(image)
    first, _ = scheduler.schedule()

    def complete(batch, token=42):
        scheduler.postprocess(
            list(scheduler.running),
            ScheduledBatchOutput(
                req_ids=list(batch.req_ids),
                token_ids=[(token,) for _ in batch.req_ids],
                num_rejected=None,
                num_bonus=None,
                draft_token_ids=None,
            ),
            batch=batch,
        )

    complete(first)
    for _ in range(9):
        decode, _ = scheduler.schedule()
        complete(decode)
    retained = list(image.token_ids)
    assert len(retained) == 18  # Original prompt plus ten generated tokens.
    scheduler.running.remove(image)
    assert scheduler.preempt(image)
    follower = seq_factory([10, 11, 12, 13])
    scheduler.add(follower)

    prompt, _ = scheduler.schedule()
    assert list(prompt.scheduled_tokens) == retained[:8]
    assert prompt.multimodal_data[image.id] is pixels
    assert not prompt.is_final_chunk[0]
    complete(prompt, token=99)
    assert list(image.token_ids) == retained  # Discard intermediate samples.
    assert image.is_partial_prefill

    suffix, _ = scheduler.schedule()
    assert list(suffix.scheduled_tokens) == retained[8:16]
    assert not suffix.multimodal_data
    complete(suffix, token=99)
    assert list(image.token_ids) == retained

    final, _ = scheduler.schedule()
    assert list(final.scheduled_tokens) == retained[16:] + list(follower.token_ids)
    assert not final.multimodal_data
    assert all(final.is_final_chunk)
    complete(final)
    assert list(image.token_ids) == retained + [42]
    assert not image.is_partial_prefill
    assert follower not in scheduler.waiting
    assert scheduler._partial_prefill_count == 0
