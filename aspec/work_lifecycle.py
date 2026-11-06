# Copyright 2026 Ant Group Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime ledger for scheduler identity and transferred-output ownership."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

WORK_ID_KEY = "aspec_work_id"
EXECUTION_ID_KEY = "aspec_execution_id"
WORK_KIND_KEY = "aspec_work_kind"
WORK_IDENTITY_KEYS = (WORK_ID_KEY, EXECUTION_ID_KEY, WORK_KIND_KEY)


class WorkLifecycleError(RuntimeError):
    """An output violated the scheduler or ownership state machine."""


def build_work_identity(*, split: str, epoch: int, source_id: str) -> dict[str, str]:
    """Build a deterministic semantic ID and the first execution-attempt ID."""

    payload = json.dumps(
        {"epoch": int(epoch), "source_id": str(source_id), "split": split},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    work_id = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return {
        WORK_ID_KEY: work_id,
        EXECUTION_ID_KEY: f"{work_id}:attempt:0",
        WORK_KIND_KEY: split,
    }


def require_work_identity(
    metadata: Mapping[str, Any] | None,
) -> tuple[str, str, str]:
    """Return the scheduler-owned identity or fail at the canonical boundary."""

    metadata = metadata or {}
    work_id = metadata.get(WORK_ID_KEY)
    execution_id = metadata.get(EXECUTION_ID_KEY)
    work_kind = metadata.get(WORK_KIND_KEY)
    if not isinstance(work_id, str) or not work_id:
        raise WorkLifecycleError(f"missing non-empty {WORK_ID_KEY}")
    if not isinstance(execution_id, str) or not execution_id:
        raise WorkLifecycleError(f"missing non-empty {EXECUTION_ID_KEY}")
    if not isinstance(work_kind, str) or not work_kind:
        raise WorkLifecycleError(f"missing non-empty {WORK_KIND_KEY}")
    return work_id, execution_id, work_kind


@dataclass
class _Record:
    execution_id: str
    work_id: str
    work_kind: str
    data_id: str
    mooncake_key: str
    size_bytes: int
    state: str = "produced"
    expected_consumers: int = 0
    released_by: set[str] = field(default_factory=set)
    release_outcomes: dict[str, str] = field(default_factory=dict)
    release_outcome: str | None = None
    optimizer_step: int | None = None


class WorkLifecycleLedger:
    """Track one owner and legal terminal transitions for generated outputs.

    This object is intentionally local to the controller actor.  It does not
    prescribe a distributed protocol; it validates the protocol exercised by
    the retained TorchSpec-backed path.
    """

    def __init__(self) -> None:
        self._records: dict[str, _Record] = {}
        self._key_to_execution: dict[str, str] = {}
        self._commit_steps: dict[int, tuple[str, ...]] = {}
        self._errors: list[str] = []
        self._resident_outputs = 0
        self._resident_bytes = 0
        self._peak_resident_outputs = 0
        self._peak_resident_bytes = 0

    @staticmethod
    def _identity(metadata: Mapping[str, Any] | None) -> tuple[str, str, str]:
        return require_work_identity(metadata)

    def _fail(self, message: str) -> None:
        self._errors.append(message)
        raise WorkLifecycleError(message)

    def register_output(
        self,
        *,
        data_id: str,
        mooncake_key: str,
        metadata: Mapping[str, Any] | None,
        size_bytes: int,
    ) -> str:
        work_id, execution_id, work_kind = self._identity(metadata)
        if execution_id in self._records:
            self._fail(f"duplicate execution_id: {execution_id}")
        if mooncake_key in self._key_to_execution:
            self._fail(f"duplicate live mooncake_key: {mooncake_key}")
        if size_bytes < 0:
            self._fail(f"negative output size for {execution_id}: {size_bytes}")

        record = _Record(
            execution_id=execution_id,
            work_id=work_id,
            work_kind=work_kind,
            data_id=str(data_id),
            mooncake_key=str(mooncake_key),
            size_bytes=int(size_bytes),
        )
        self._records[execution_id] = record
        self._key_to_execution[record.mooncake_key] = execution_id
        self._resident_outputs += 1
        self._resident_bytes += record.size_bytes
        self._peak_resident_outputs = max(
            self._peak_resident_outputs, self._resident_outputs
        )
        self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)
        return execution_id

    def register_outputs(self, outputs: Sequence[Mapping[str, Any]]) -> list[str]:
        """Atomically register one inference result batch.

        Validation happens for the complete batch before residency changes.  A
        malformed or duplicate later result therefore cannot leave an earlier
        result registered without being visible in the controller pool.
        """

        prepared: list[tuple[str, str, str, str, str, int]] = []
        execution_ids: set[str] = set()
        mooncake_keys: set[str] = set()
        for output in outputs:
            metadata = output.get("metadata")
            work_id, execution_id, work_kind = self._identity(metadata)
            mooncake_key = str(output["mooncake_key"])
            size_bytes = int(output["size_bytes"])
            if execution_id in self._records or execution_id in execution_ids:
                self._fail(f"duplicate execution_id: {execution_id}")
            if mooncake_key in self._key_to_execution or mooncake_key in mooncake_keys:
                self._fail(f"duplicate live mooncake_key: {mooncake_key}")
            if size_bytes < 0:
                self._fail(f"negative output size for {execution_id}: {size_bytes}")
            execution_ids.add(execution_id)
            mooncake_keys.add(mooncake_key)
            prepared.append(
                (
                    str(output["data_id"]),
                    mooncake_key,
                    work_id,
                    execution_id,
                    work_kind,
                    size_bytes,
                )
            )

        for data_id, mooncake_key, work_id, execution_id, work_kind, size_bytes in prepared:
            record = _Record(
                execution_id=execution_id,
                work_id=work_id,
                work_kind=work_kind,
                data_id=data_id,
                mooncake_key=mooncake_key,
                size_bytes=size_bytes,
            )
            self._records[execution_id] = record
            self._key_to_execution[mooncake_key] = execution_id
            self._resident_outputs += 1
            self._resident_bytes += size_bytes

        self._peak_resident_outputs = max(
            self._peak_resident_outputs, self._resident_outputs
        )
        self._peak_resident_bytes = max(self._peak_resident_bytes, self._resident_bytes)
        return [item[3] for item in prepared]

    def mark_dispatched(
        self, metadata: Mapping[str, Any] | None, *, expected_consumers: int
    ) -> None:
        _, execution_id, _ = self._identity(metadata)
        record = self._records.get(execution_id)
        if record is None:
            self._fail(f"dispatch references unknown execution_id: {execution_id}")
        if record.state != "produced":
            self._fail(
                f"cannot dispatch {execution_id} from state {record.state!r}"
            )
        if expected_consumers <= 0:
            self._fail(
                f"expected_consumers must be positive for {execution_id}, got "
                f"{expected_consumers}"
            )
        record.state = "dispatched"
        record.expected_consumers = int(expected_consumers)

    def release_output(
        self,
        *,
        execution_id: str,
        data_id: str,
        mooncake_key: str,
        participant: str,
        outcome: str,
    ) -> None:
        record = self._records.get(execution_id)
        if record is None:
            self._fail(f"release references unknown execution_id: {execution_id}")
        if record.data_id != str(data_id) or record.mooncake_key != str(mooncake_key):
            self._fail(f"release identity mismatch for {execution_id}")
        if record.state != "dispatched":
            self._fail(f"cannot release {execution_id} from state {record.state!r}")
        if participant in record.released_by:
            self._fail(f"duplicate release of {execution_id} by {participant}")
        if outcome not in {"consumed", "discarded_after_error"}:
            self._fail(f"invalid release outcome for {execution_id}: {outcome!r}")

        record.released_by.add(participant)
        record.release_outcomes[participant] = outcome
        if len(record.released_by) > record.expected_consumers:
            self._fail(f"too many releases for {execution_id}")
        if len(record.released_by) < record.expected_consumers:
            return

        if "discarded_after_error" in record.release_outcomes.values():
            record.release_outcome = "discarded_after_error"
            record.state = "failed"
        elif record.work_kind == "train":
            record.release_outcome = "consumed"
            record.state = "released"
        else:
            record.release_outcome = "consumed"
            record.state = "completed"
        self._drop_residency(record)

    def begin_discard(self, metadata: Mapping[str, Any] | None) -> None:
        _, execution_id, _ = self._identity(metadata)
        record = self._records.get(execution_id)
        if record is None:
            self._fail(f"discard references unknown execution_id: {execution_id}")
        if record.state != "produced":
            self._fail(f"cannot discard {execution_id} from state {record.state!r}")
        record.state = "discard_pending"

    def confirm_discarded(self, metadata_items: Sequence[Mapping[str, Any] | None]) -> None:
        for metadata in metadata_items:
            _, execution_id, _ = self._identity(metadata)
            record = self._records.get(execution_id)
            if record is None:
                self._fail(f"discard confirmation references unknown execution_id: {execution_id}")
            if record.state != "discard_pending":
                self._fail(
                    f"cannot confirm discard of {execution_id} from state {record.state!r}"
                )
            record.state = "discarded"
            record.release_outcome = "unused_output_discarded"
            self._drop_residency(record)

    def commit_optimizer_step(
        self,
        *,
        optimizer_step: int,
        execution_ids: Sequence[str],
        expected_unique: int,
        expected_occurrences: int = 1,
    ) -> None:
        if optimizer_step in self._commit_steps:
            self._fail(f"optimizer step already committed: {optimizer_step}")
        counts = Counter(execution_ids)
        if len(counts) != expected_unique:
            self._fail(
                f"optimizer step {optimizer_step} has {len(counts)} unique work items; "
                f"expected {expected_unique}"
            )
        wrong = {
            execution_id: count
            for execution_id, count in counts.items()
            if count != expected_occurrences
        }
        if wrong:
            self._fail(
                f"optimizer step {optimizer_step} has invalid work multiplicity: {wrong}"
            )
        for execution_id in counts:
            record = self._records.get(execution_id)
            if record is None:
                self._fail(
                    f"optimizer step {optimizer_step} references unknown work: {execution_id}"
                )
            if record.state != "released":
                self._fail(
                    f"optimizer step {optimizer_step} cannot commit {execution_id} "
                    f"from state {record.state!r}"
                )
        committed = tuple(sorted(counts))
        for execution_id in committed:
            record = self._records[execution_id]
            record.state = "committed"
            record.optimizer_step = int(optimizer_step)
        self._commit_steps[int(optimizer_step)] = committed

    def discard_released_uncommitted(self) -> int:
        """Finalize cancelled training work after every consumer released storage.

        A trainer may release all transfer ownership before the optimizer step is
        committed.  On cooperative shutdown that work is intentionally eligible
        for regeneration, but leaving it in ``released`` makes the semantic state
        ambiguous.  Only storage-free training records can cross this boundary;
        dispatched or resident outputs remain incomplete and visible as errors.
        """

        discarded = 0
        for record in self._records.values():
            if record.work_kind == "train" and record.state == "released":
                record.state = "discarded"
                record.release_outcome = "uncommitted_work_discarded"
                discarded += 1
        return discarded

    def discard_after_owner_fence(self, execution_ids: Sequence[str]) -> int:
        """Finalize owned work only after the caller fenced every read process.

        This does not fence processes itself. Shared-source retirement calls it
        after all source read rights were withdrawn and consumer owners exited.
        Committed work is never rewritten; repeated retirement is idempotent.
        """
        records = []
        for identity in execution_ids:
            if identity not in self._records:
                self._fail(f"fenced discard references unknown work: {identity}")
            records.append(self._records[identity])
        discarded = 0
        for record in records:
            if record.state in {"committed", "completed", "discarded", "failed"}:
                continue
            if record.state not in {"produced", "dispatched", "discard_pending", "released"}:
                self._fail(f"cannot fenced-discard {record.execution_id} from {record.state}")
            if record.state != "released":
                self._drop_residency(record)
            record.state = "discarded"
            record.release_outcome = "owners_fenced_read_right_withdrawn"
            discarded += 1
        return discarded

    def discard_all_after_owner_fence(self) -> int:
        """Retire the failed attempt after its owner failure domains are fenced."""
        return self.discard_after_owner_fence(tuple(self._records))

    def _drop_residency(self, record: _Record) -> None:
        self._resident_outputs -= 1
        self._resident_bytes -= record.size_bytes
        self._key_to_execution.pop(record.mooncake_key, None)
        if self._resident_outputs < 0 or self._resident_bytes < 0:
            self._fail("lifecycle residency accounting became negative")

    @property
    def resident_outputs(self) -> int:
        return self._resident_outputs

    @property
    def resident_bytes(self) -> int:
        return self._resident_bytes

    def committed_work_ids(self) -> list[str]:
        """Return the semantic work set that reached optimizer commit."""

        work_ids = [
            record.work_id
            for record in self._records.values()
            if record.state == "committed"
        ]
        if len(work_ids) != len(set(work_ids)):
            self._fail("multiple committed executions reference the same semantic work ID")
        return sorted(work_ids)

    def status(self) -> dict[str, Any]:
        state_counts = Counter(record.state for record in self._records.values())
        produced = len(self._records)
        committed = state_counts["committed"]
        discarded = state_counts["discarded"]
        failed = state_counts["failed"]
        completed = state_counts["completed"]
        incomplete = produced - committed - discarded - failed - completed
        optimizer_steps = sorted(self._commit_steps)
        contiguous_steps = not optimizer_steps or optimizer_steps == list(
            range(optimizer_steps[0], optimizer_steps[-1] + 1)
        )
        valid = (
            not self._errors
            and self._resident_outputs == 0
            and self._resident_bytes == 0
            and incomplete == 0
            and contiguous_steps
        )
        return {
            "state": "validated" if valid else "incomplete",
            "produced_outputs": produced,
            "committed_work": committed,
            "discarded_outputs": discarded,
            "completed_non_training_outputs": completed,
            "failed_outputs": failed,
            "incomplete_outputs": incomplete,
            "resident_outputs": self._resident_outputs,
            "resident_bytes": self._resident_bytes,
            "peak_resident_outputs": self._peak_resident_outputs,
            "peak_resident_bytes": self._peak_resident_bytes,
            "optimizer_steps_committed": len(self._commit_steps),
            "optimizer_steps_contiguous": contiguous_steps,
            "state_counts": dict(sorted(state_counts.items())),
            "errors": list(self._errors),
        }
