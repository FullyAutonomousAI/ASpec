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

"""Check local file links in maintained public Markdown."""

import argparse
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit


def check_links(root):
    root = Path(root).resolve()
    files = list(root.glob("*.md"))
    for directory in ("docs", "configs", "tools", "examples", "patches", ".github"):
        files.extend((root / directory).rglob("*.md"))
    errors = []
    for path in sorted(files):
        relative = path.relative_to(root)
        fenced = False
        for line_number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            for target in re.findall(r"\[[^\]\n]*\]\(([^)]+)\)", line):
                target = target.strip().split(' "', 1)[0].strip("<>")
                if urlsplit(target).scheme or target.startswith("#"):
                    continue
                local = unquote(target.split("#", 1)[0])
                if local and not (path.parent / local).exists():
                    errors.append((str(relative), line_number, target))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check_links(args.root)
    for path, line, target in errors:
        print(f"{path}:{line}: missing local target: {target}")
    if not errors:
        print("PASS maintained Markdown local file links (external URLs/anchors not checked)")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
