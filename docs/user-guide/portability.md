# Environment portability

Recipes describe a workload and required capabilities. They do not identify a particular developer
host. Use [installation](installation.md), [recipe inputs](recipes.md) and [multi-node setup](multinode.md)
to bind them to your environment.

| Item | Public contract |
| --- | --- |
| Host/address | User-supplied Ray/node IPs and SSH aliases; no bundled private hostnames |
| Model/data | Declared model family plus explicit local mirror/Hub ID; absolute train/eval inputs; tokenizer compatibility is mandatory |
| Source/venv | Each node installs matching source/packages/patches independently; no import from another checkout through PYTHONPATH |
| CUDA libraries | The wrapper discovers the active venv's libraries; optional private compatibility directory is explicit; no system driver changes |
| GPU IDs | `CUDA_VISIBLE_DEVICES` selects allocated devices for single-node acceptance; recipes specify a resource budget, not fixed physical IDs |
| Hardware capability | NVLink/precision/attention requirements are checked; a marketing name or GPU count does not prove them |
| Network interface | Default live NCCL/Gloo selection; optional intentional interface overrides; no fixed `eth0`/RDMA device in the release recipe |
| Ports/authentication | User selects available Ray endpoints and consistent cluster credentials; task-owned local smoke uses isolated Ray state |
| Writable output | New per-run directory; no implicit overwrite of existing ASpec state |
| Multi-node disks | Independent local filesystems by default; input replicas verified, checkpoint shards collected and resume state distributed |
| Logical input layout | One multi-node config currently uses the same absolute input/output path strings across nodes; no per-node path remapping yet |
| Caches | Each node owns its loader/JIT caches; multi-node workers retain their startup environment instead of the driver's filesystem paths |
| Native prerequisites | Supported Linux/Python/CUDA lane, compatible compiler and required RDMA libraries remain explicit installation prerequisites |

`tools.stage_recipe_inputs` copies and verifies explicit inputs over SSH when needed. It does not
assume NFS, download a model without a chosen source, overwrite an existing destination, or install a
remote environment. `tools/install.sh` handles the pinned Python lane and training patch after native
prerequisites are available. Run it independently in each node's activated venv.

Examples containing `/models`, `/data`, node aliases or GPU lists show syntax, not provisioned resources.
Comparison configs under `configs/benchmarks/` may use upstream `/path/to/...` placeholders; they are
separate benchmark inputs and must be bound explicitly. Supplementary shared-storage configs opt into
the shared contract. Machine-specific records are not part of the source release.

CPU validation checks configuration, identities and negative paths. Actual memory fit, transport,
collectives and worker startup require GPU/runtime evidence. See [acceptance](../contributing/validation.md).
