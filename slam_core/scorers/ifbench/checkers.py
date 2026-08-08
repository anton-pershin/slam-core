# Copyright 2025 Allen Institute for AI.
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

"""Factories for IFBench instructions."""

from __future__ import annotations

from slam_core.scorers.ifbench.checker_factory import IFBenchCheckerFactory
from slam_core.scorers.ifbench.instructions_registry import INSTRUCTION_DICT

__all__ = ["build_checker_factory"]


def build_checker_factory() -> IFBenchCheckerFactory:
    factory = IFBenchCheckerFactory()
    for instruction_id, checker_cls in INSTRUCTION_DICT.items():
        factory.register(instruction_id, checker_cls)
    return factory
