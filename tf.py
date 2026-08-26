# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
本文件属于 Alpamayo 项目。
主要功能：负责空间坐标系的转换（类似 ROS 里的 tf 树），用来算出相机、车身和真实世界之间的位置关系。

【与其他文件的依赖调用关系】
- 向上依赖（本文件调用了谁）：
  1. 调用了 `interpolation.py`：如果在坐标系转换时遇到时间对应不上的情况，就用它来插值算出精确时间的坐标。
- 向下提供（谁调用了本文件）：
  1. 被 `egomotion.py` 继承和调用：让车辆自身的运动状态数据，具备能够在不同坐标系下被无缝转换的能力。
"""

import dataclasses
import enum
from typing import ClassVar
from typing_extensions import Self

import numpy as np
import scipy.spatial.transform as spt

import interpolation


@dataclasses.dataclass(frozen=True)
class FrameInfo:
    frame_id: str
    timestamp: int | str | None = None

    def __lt__(self, other: Self) -> bool:
        if self.frame_id != other.frame_id:
            raise ValueError(
                f"Nonmatching {self.frame_id=} and {other.frame_id=} cannot be compared."
            )
        if not isinstance(self.timestamp, int) or not isinstance(other.timestamp, int):
            raise ValueError(
                f"{self.frame_id=} and {other.frame_id=} cannot be compared; "
                "only integer timestamps can be compared."
            )
        return self.timestamp < other.timestamp


@dataclasses.dataclass(frozen=True)
class FrameTransform:
    target_frame_info: FrameInfo
    source_frame_info: FrameInfo
    tf_target_source: spt.RigidTransform

    def check(self, frame_info: FrameInfo) -> None:
        if self.source_frame_info != frame_info:
            raise ValueError(f"{self.source_frame_info=} does not match {frame_info=}.")


class TransformTree:

    def __init__(self, root_frame_id: str = "anchor") -> None:
        self.parent = {}
        self.root_frame_id = root_frame_id

    def lookup_transform(
        self,
        target_frame_info: FrameInfo,
        source_frame_info: FrameInfo,
    ) -> FrameTransform:

        tf_root_target = self._compute_tf_root_frame(target_frame_info)
        tf_root_source = self._compute_tf_root_frame(source_frame_info)

        return FrameTransform(
            target_frame_info,
            source_frame_info,
            tf_root_target.inv() @ tf_root_source,
        )

    def add_transform(
        self,
        parent_frame_id: str,
        child_frame_id: str,
        tf_parent_child,
    ) -> None:

        if child_frame_id in self.parent:
            raise ValueError(f"{child_frame_id=} is already in the transform tree.")
        self.parent[child_frame_id] = (parent_frame_id, tf_parent_child)

    def _compute_tf_root_frame(self, frame_info: FrameInfo):
        frame_id = frame_info.frame_id
        tf_root_frame = spt.RigidTransform.identity()

        while frame_id != self.root_frame_id:
            frame_id, tf_parent_frame = self.parent[frame_id]

            if not isinstance(tf_parent_frame, spt.RigidTransform):
                if isinstance(tf_parent_frame, interpolation.RigidTransformInterpolator):
                    tf_parent_frame = tf_parent_frame(frame_info.timestamp)
                else:
                    raise ValueError(f"Unknown transform type {type(tf_parent_frame)=}.")

            tf_root_frame = tf_parent_frame @ tf_root_frame

        return tf_root_frame


# ✅ 修复 StrEnum
class TransformableType(str, enum.Enum):
    POINT = "point"
    POSE = "pose"
    VECTOR = "vector"


class Transformable:

    TRANSFORMABLE_TYPE_KEY: ClassVar[str] = "transformable_type"

    POINT: ClassVar[dict[str, TransformableType]] = {
        TRANSFORMABLE_TYPE_KEY: TransformableType.POINT
    }

    POSE: ClassVar[dict[str, TransformableType]] = {
        TRANSFORMABLE_TYPE_KEY: TransformableType.POSE
    }

    VECTOR: ClassVar[dict[str, TransformableType]] = {
        TRANSFORMABLE_TYPE_KEY: TransformableType.VECTOR
    }

    def transform(self, rigid_transform: spt.RigidTransform) -> Self:

        def _transform_field(field):
            field_value = getattr(self, field.name)

            if isinstance(field_value, Transformable):
                return field_value.transform(rigid_transform)

            transformable_type = field.metadata.get(
                Transformable.TRANSFORMABLE_TYPE_KEY
            )

            if transformable_type == TransformableType.POINT:
                if not isinstance(field_value, np.ndarray):
                    raise ValueError(
                        f"Expected {field.name} to be ndarray, got {type(field_value)}."
                    )
                return rigid_transform.apply(field_value)

            elif transformable_type == TransformableType.POSE:
                if not isinstance(field_value, spt.RigidTransform):
                    raise ValueError(
                        f"Expected {field.name} to be RigidTransform, got {type(field_value)}."
                    )
                return rigid_transform * field_value

            elif transformable_type == TransformableType.VECTOR:
                if not isinstance(field_value, np.ndarray):
                    raise ValueError(
                        f"Expected {field.name} to be ndarray, got {type(field_value)}."
                    )
                return rigid_transform.rotation.apply(field_value)

            return field_value

        return dataclasses.replace(
            self,
            **{
                field.name: _transform_field(field)
                for field in dataclasses.fields(self)
                if (
                    isinstance(getattr(self, field.name), Transformable)
                    or Transformable.TRANSFORMABLE_TYPE_KEY in field.metadata
                )
            },
        )

    def transform_frame(self, frame_transform: FrameTransform, skip_check: bool = False) -> Self:

        if not skip_check:
            frame_transform.check(self.frame_info)

        return dataclasses.replace(
            self.transform(frame_transform.tf_target_source),
            frame_info=frame_transform.target_frame_info,
        )