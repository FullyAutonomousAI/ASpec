# Copyright 2026 Ant Group Co., Ltd.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locate installed assets, or their canonical files in a source checkout."""

from pathlib import Path, PurePosixPath

PACKAGE_ROOT = Path(__file__).resolve().parent


def asset_root():
    installed = PACKAGE_ROOT / "resources"
    if installed.is_dir():
        return installed
    source = PACKAGE_ROOT.parent
    if (source / "pyproject.toml").is_file():
        return source
    raise FileNotFoundError("ASpec resources are missing; reinstall the wheel")


def asset_path(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("ASpec assets require a package-relative path")
    if path.parts[:2] == ("aspec", "config"):
        result = PACKAGE_ROOT.joinpath(*path.parts[1:])
    else:
        result = asset_root().joinpath(*path.parts)
    if not result.is_file():
        raise FileNotFoundError(f"Missing ASpec asset: {relative}")
    return result


def recipe_config(relative):
    """Resolve bundled input assets without moving outputs into site-packages."""
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(asset_path(relative)), resolve=False)

    def resolve(value):
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        if isinstance(value, str):
            value = value.replace("aspec/config/", str(PACKAGE_ROOT / "config") + "/")
            value = value.replace("../../examples/data/sample_conversations.jsonl",
                                  str(asset_path("examples/data/sample_conversations.jsonl")))
        return value

    return OmegaConf.create(resolve(config))
