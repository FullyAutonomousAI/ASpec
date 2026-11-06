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

"""Copy or verify an explicit input on SSH hosts without a shared filesystem."""

import argparse
import hashlib
import shlex
import subprocess
import uuid
from pathlib import Path


def checksum_lines(source):
    source = Path(source).resolve()
    paths = sorted(p for p in source.rglob("*") if p.is_file()) if source.is_dir() else [source]
    if not paths:
        raise ValueError("Input must contain files")
    lines = []
    for path in paths:
        name = path.relative_to(source).as_posix() if source.is_dir() else source.name
        if "\n" in name or "\\" in name:
            raise ValueError("Input filenames with newlines/backslashes are unsupported")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(data)
        lines.append(f"{digest.hexdigest()}  {name}\n")
    return "".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", required=True, help="Same absolute path on each host")
    parser.add_argument(
        "--host", action="append", required=True, help="SSH host/alias; repeat per node"
    )
    parser.add_argument("--ssh-config", help="Optional SSH config file")
    parser.add_argument(
        "--verify-only", action="store_true", help="Verify existing copies without writing"
    )
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    destination = Path(args.destination)
    if not destination.is_absolute() or destination == Path("/"):
        parser.error("Choose an explicit absolute destination below the filesystem root")
    if not source.is_dir() and source.name != destination.name:
        parser.error("File inputs retain their basename; use a destination with the same name")
    checksums = checksum_lines(source)
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if args.ssh_config:
        ssh += ["-F", str(Path(args.ssh_config).resolve())]
    for host in args.host:
        if host.startswith("-") or any(c.isspace() for c in host):
            parser.error("Invalid SSH host/alias")
        if not args.verify_only:
            temporary = str(destination) + ".stage-" + uuid.uuid4().hex
            q = shlex.quote
            command = (
                f"test ! -e {q(str(destination))} && test ! -L {q(str(destination))} && mkdir -p {q(str(destination.parent))} && "
                f"mkdir {q(temporary)} && tar -xf - -C {q(temporary)} && "
                f"mv -T -n {q(temporary + '/' + source.name)} {q(str(destination))} && test ! -e {q(temporary + '/' + source.name)} && rmdir {q(temporary)}"
            )
            producer = subprocess.Popen(
                ["tar", "-chf", "-", "-C", str(source.parent), "--", source.name],
                stdout=subprocess.PIPE,
            )
            try:
                copied = subprocess.run([*ssh, host, command], stdin=producer.stdout, check=False)
            finally:
                producer.stdout.close()
                code = producer.wait()
            if copied.returncode or code:
                raise RuntimeError(
                    f"Input copy failed for {host}; existing destinations are never overwritten. Inspect {temporary}"
                )
        directory = destination if source.is_dir() else destination.parent
        subprocess.run(
            [*ssh, host, f"cd {shlex.quote(str(directory))} && sha256sum -c -"],
            input=checksums,
            text=True,
            check=True,
        )
        print(f"Verified {host}:{destination}", flush=True)


if __name__ == "__main__":
    main()
