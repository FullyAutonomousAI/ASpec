# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shared Target-prefill concurrency and token-capacity calculations."""


def effective_engine_concurrency(
    *,
    configured: int,
    batch_size: int,
    source_slots: int | None,
    slots_per_sample: int = 1,
) -> int:
    """Return the requests per engine that can coexist in the source ring."""
    if configured < 1 or batch_size < 1 or slots_per_sample < 1:
        raise ValueError(
            "configured concurrency, batch size, and slots per sample must be positive"
        )
    if source_slots is None:
        return configured
    slots_per_request = batch_size * slots_per_sample
    if source_slots < slots_per_request:
        raise ValueError(
            "source ring cannot hold one inference request: "
            f"source_slots={source_slots}, require at least {slots_per_request} for "
            f"batch_size={batch_size}, slots_per_sample={slots_per_sample}"
        )
    return min(configured, source_slots // slots_per_request)


def direct_output_sample_capacity(
    *, engine_count: int, source_slots: int, slots_per_sample: int = 1
) -> int:
    """Return how many direct-output samples can remain resident at once.

    A Target engine may own more than one live sample as long as its
    generation-checked source ring has enough slots.  Source diversity helps
    balance releases, but one distinct engine per Draft rank is not required.
    """
    if engine_count < 1 or source_slots < 1 or slots_per_sample < 1:
        raise ValueError(
            "engine count, source slots, and slots per sample must be positive"
        )
    return engine_count * (source_slots // slots_per_sample)


def required_target_prefill_tokens(
    *, max_sequence_length: int, batch_size: int, max_concurrent_batches: int
) -> int:
    """Return the SGLang token pool needed by concurrent no-decode prefills."""
    if max_sequence_length < 1:
        raise ValueError("max sequence length must be positive")
    return max_sequence_length * batch_size * max_concurrent_batches + 256
