from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import ConfigError


LASER_ORIENTATIONS = frozenset({"horizontal", "vertical"})


def normalize_laser_orientation(
    value: Any,
    *,
    field_name: str = "laser.orientation",
) -> str:
    if not isinstance(value, str) or value not in LASER_ORIENTATIONS:
        allowed = ", ".join(sorted(LASER_ORIENTATIONS))
        raise ConfigError(f"{field_name} 必须是 {allowed} 之一，当前值：{value!r}")
    return value


@dataclass(frozen=True, slots=True)
class LaserConfig:
    orientation: str = "horizontal"

    def __post_init__(self) -> None:
        object.__setattr__(self, "orientation", normalize_laser_orientation(self.orientation))


def parse_laser_config(value: Any, *, field_name: str = "laser") -> LaserConfig:
    if value is None:
        return LaserConfig()
    if not isinstance(value, Mapping):
        raise ConfigError(f"{field_name} 必须是映射")
    unknown = set(value) - {"orientation"}
    if unknown:
        raise ConfigError(f"{field_name} 包含未知字段：{sorted(unknown)}")
    return LaserConfig(
        orientation=normalize_laser_orientation(
            value.get("orientation", "horizontal"),
            field_name=f"{field_name}.orientation",
        )
    )


__all__ = [
    "LASER_ORIENTATIONS",
    "LaserConfig",
    "normalize_laser_orientation",
    "parse_laser_config",
]
