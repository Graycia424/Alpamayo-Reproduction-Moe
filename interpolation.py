# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""
本文件属于 Alpamayo 项目。
主要功能：一个纯数学工具，用来做数据的平滑插值（比如把每秒10帧的数据补齐到每秒30帧，或者把两个坐标点中间的轨迹算出来）。

【与其他文件的依赖调用关系】
- 向上依赖（本文件调用了谁）：
  它是一个底层的数学工具包，没有调用项目里的其他兄弟文件，主要是调了 `scipy.interpolate` 库。
- 向下提供（谁调用了本文件）：
  1. 被 `egomotion.py` 调用：专门用来给车辆的位置、旋转姿态和速度数据做时间上的对齐和补帧。
  2. 被 `tf.py` 调用：在做坐标系变换时，如果遇到时间对不上的情况就用它插值计算。
"""

import dataclasses
import enum
from typing import Callable, ClassVar, TypeVar, Generic

import numpy as np
import numpy.typing as npt
import scipy.interpolate as spi
import scipy.spatial.transform as spt


class InterpolationMethod(str, enum.Enum):
    """Supported interpolation methods for vectors and rotations."""

    LINEAR = "linear"
    CUBIC_SPLINE = "cubic_spline"
    SLERP = "slerp"
    ROTATION_SPLINE = "rotation_spline"


@dataclasses.dataclass
class RigidTransformInterpolationMethod:
    """Specifies an interpolation method for `RigidTransform`s."""

    rotation: InterpolationMethod = InterpolationMethod.SLERP
    translation: InterpolationMethod = InterpolationMethod.CUBIC_SPLINE


@dataclasses.dataclass
class RigidTransformInterpolator:
    """Interpolator for `RigidTransform`s."""

    timestamps: npt.NDArray[np.int64]
    values: spt.RigidTransform
    interpolation_method: RigidTransformInterpolationMethod = dataclasses.field(
        default_factory=RigidTransformInterpolationMethod
    )

    def __post_init__(self) -> None:
        self.relative_timestamps = self.timestamps - self.timestamps[0]

        self.interpolants = {
            "rotation": create_interpolant(
                self.interpolation_method.rotation,
                self.relative_timestamps,
                self.values.rotation,
            ),
            "translation": create_interpolant(
                self.interpolation_method.translation,
                self.relative_timestamps,
                self.values.translation,
            ),
        }

    def __call__(self, timestamp: npt.ArrayLike) -> spt.RigidTransform:
        return spt.RigidTransform.from_components(
            rotation=self.interpolants["rotation"](timestamp - self.timestamps[0]),
            translation=self.interpolants["translation"](timestamp - self.timestamps[0]),
        )


def create_interpolant(
    interpolation_method,
    timestamps: npt.NDArray[np.int64],
    values,
) -> Callable:
    """Creates an interpolator (with extrapolation disallowed)."""

    if interpolation_method == InterpolationMethod.LINEAR:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected ndarray, got {type(values)}.")
        linear = spi.make_interp_spline(timestamps, values, k=1)
        linear.extrapolate = False
        return linear

    elif interpolation_method == InterpolationMethod.CUBIC_SPLINE:
        if not isinstance(values, np.ndarray):
            raise ValueError(f"Expected ndarray, got {type(values)}.")
        return spi.CubicSpline(timestamps, values, extrapolate=False)

    elif interpolation_method == InterpolationMethod.SLERP:
        if not isinstance(values, spt.Rotation):
            raise ValueError(f"Expected Rotation, got {type(values)}.")
        return spt.Slerp(timestamps, values)

    elif interpolation_method == InterpolationMethod.ROTATION_SPLINE:
        if not isinstance(values, spt.Rotation):
            raise ValueError(f"Expected Rotation, got {type(values)}.")
        return spt.RotationSpline(timestamps, values)

    elif isinstance(interpolation_method, RigidTransformInterpolationMethod):
        if not isinstance(values, spt.RigidTransform):
            raise ValueError(f"Expected RigidTransform, got {type(values)}.")
        return RigidTransformInterpolator(timestamps, values, interpolation_method)

    raise ValueError(f"Unknown interpolation method: {interpolation_method}")


# ✅ 关键修复：改为 Python 3.10 兼容泛型写法
T = TypeVar("T", bound="Interpolatable")


@dataclasses.dataclass
class Interpolator(Generic[T]):
    """Interpolator for dataclasses with interpolation method specified by field metadata."""

    timestamps: npt.NDArray[np.int64]
    values: T

    def __post_init__(self) -> None:
        self.value_type = type(self.values)
        self.relative_timestamps = self.timestamps - self.timestamps[0]

        def _create_interpolant(field):
            field_values = getattr(self.values, field.name)

            if isinstance(field_values, Interpolatable):
                return field_values.create_interpolator(self.relative_timestamps)

            if Interpolatable.INTERPOLATION_METHOD_KEY not in field.metadata:
                raise ValueError(
                    f"Missing interpolation method for "
                    f"{self.value_type.__name__}.{field.name}."
                )

            interpolation_method = field.metadata[
                Interpolatable.INTERPOLATION_METHOD_KEY
            ]

            return create_interpolant(
                interpolation_method,
                self.relative_timestamps,
                field_values,
            )

        self.interpolants = {
            field.name: _create_interpolant(field)
            for field in dataclasses.fields(self.value_type)
        }

    def __call__(self, timestamp: npt.ArrayLike) -> T:
        return self.value_type(
            **{
                field.name: self.interpolants[field.name](
                    timestamp - self.timestamps[0]
                )
                for field in dataclasses.fields(self.value_type)
            }
        )

    @property
    def time_range(self) -> tuple[int, int]:
        return self.timestamps[0], self.timestamps[-1]

    def __repr__(self) -> str:
        return (
            f"Interpolator[{self.value_type.__name__}]"
            f"(time_range=[{self.time_range}])"
        )


class Interpolatable:
    """Base class for enabling interpolatable dataclasses."""

    INTERPOLATION_METHOD_KEY: ClassVar[str] = "interpolation_method"

    LINEAR: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.LINEAR
    }

    CUBIC_SPLINE: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.CUBIC_SPLINE
    }

    SLERP: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.SLERP
    }

    ROTATION_SPLINE: ClassVar[dict[str, InterpolationMethod]] = {
        INTERPOLATION_METHOD_KEY: InterpolationMethod.ROTATION_SPLINE
    }

    DEFAULT_RIGID_TRANSFORM_INTERPOLATION: ClassVar[
        dict[str, RigidTransformInterpolationMethod]
    ] = {
        INTERPOLATION_METHOD_KEY: RigidTransformInterpolationMethod()
    }

    def create_interpolator(
        self, timestamps: npt.NDArray[np.int64]
    ) -> "Interpolator":
        return Interpolator(timestamps, self)