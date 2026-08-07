"""Analytical reconstruction of the three-region COMSOL moving mesh.

Reference liquid domain (R, Z):
    R in [r1, r2], where r2 = r1 + sx
    Z in [b1, h_all - b1], where h_all = 2*b1 + 2*sy + b2

COMSOL displacement rules:
    lower:  m_down   = (Z - b1) * m_middle / sy
    middle: m_middle = A * (1 - cos(2*pi*t/Ts))
    upper:  m_up     = (h_all - b1 - Z) * m_middle / sy

The mesh velocity is the analytical time derivative of these displacements.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Union

import numpy as np


class MeshRegion(IntEnum):
    LOWER = 0
    MIDDLE = 1
    UPPER = 2


@dataclass(frozen=True)
class DamperGeometry:
    r1: float
    r2: float
    r3: float
    b1: float
    b2: float
    sy: float
    h_all: float

    @classmethod
    def from_json(
        cls,
        geometry: Mapping[str, Union[int, float, str]],
        unit_scale: float = 1.0e-3,
    ) -> "DamperGeometry":
        r1 = float(geometry["r1"]) * unit_scale
        r2 = r1 + float(geometry["sx"]) * unit_scale
        r3 = r2 + float(geometry["a2"]) * unit_scale
        b1 = float(geometry["b1"]) * unit_scale
        b2 = float(geometry["b2"]) * unit_scale
        sy = float(geometry["sy"]) * unit_scale
        h_all = 2.0 * b1 + 2.0 * sy + b2
        return cls(r1=r1, r2=r2, r3=r3, b1=b1, b2=b2, sy=sy, h_all=h_all)

    @property
    def lower_top(self) -> float:
        return self.b1 + self.sy

    @property
    def middle_top(self) -> float:
        return self.b1 + self.sy + self.b2

    @property
    def fluid_top(self) -> float:
        return self.h_all - self.b1


@dataclass(frozen=True)
class MeshState:
    time: float
    pos: np.ndarray
    region: np.ndarray
    motion_weight: np.ndarray
    node_displacement: np.ndarray
    piston_displacement: float
    mesh_velocity: np.ndarray
    piston_velocity: float


def piston_displacement(t: float, amplitude: float, period: float) -> float:
    """Current COMSOL an1(t): A * (1 - cos(2*pi/Ts*t))."""
    if period <= 0.0:
        raise ValueError(f"Loading period Ts must be positive, got {period}.")
    return float(amplitude * (1.0 - np.cos(2.0 * np.pi * t / period)))


def piston_velocity(t: float, amplitude: float, period: float) -> float:
    """Time derivative of the current COMSOL an1(t)."""
    if period <= 0.0:
        raise ValueError(f"Loading period Ts must be positive, got {period}.")
    omega = 2.0 * np.pi / period
    return float(amplitude * omega * np.sin(omega * t))


def classify_mesh_regions(
    reference_pos: np.ndarray,
    geometry: DamperGeometry,
    tolerance: float = 1.0e-9,
) -> np.ndarray:
    """Classify liquid-mesh nodes by their reference Z coordinate."""
    z = np.asarray(reference_pos, dtype=np.float64)[:, 1]
    region = np.full(z.shape, MeshRegion.MIDDLE, dtype=np.int64)
    region[z < geometry.lower_top - tolerance] = MeshRegion.LOWER
    region[z > geometry.middle_top + tolerance] = MeshRegion.UPPER
    return region


def compute_motion_weights(
    reference_pos: np.ndarray,
    geometry: DamperGeometry,
    region: np.ndarray,
) -> np.ndarray:
    """Reproduce m_down, m_middle, and m_up from the COMSOL model."""
    z = np.asarray(reference_pos, dtype=np.float64)[:, 1]
    weights = np.ones(z.shape, dtype=np.float64)

    lower = region == MeshRegion.LOWER
    middle = region == MeshRegion.MIDDLE
    upper = region == MeshRegion.UPPER

    weights[lower] = (z[lower] - geometry.b1) / geometry.sy
    weights[middle] = 1.0
    weights[upper] = (geometry.h_all - geometry.b1 - z[upper]) / geometry.sy
    return np.clip(weights, 0.0, 1.0)


class DamperMeshMotion:
    """Restore the moving liquid mesh from a static reference mesh and JSON parameters."""

    def __init__(
        self,
        reference_pos: np.ndarray,
        geometry: Mapping[str, Union[int, float, str]],
        loading: Mapping[str, Union[int, float, str]],
        unit_scale: float = 1.0e-3,
        validate_domain: bool = True,
        tolerance: float = 1.0e-8,
    ) -> None:
        pos = np.asarray(reference_pos, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] < 2:
            raise ValueError(f"reference_pos must have shape [N, 2+], got {pos.shape}.")

        self.reference_pos = pos[:, :2].copy()
        self.geometry = DamperGeometry.from_json(geometry, unit_scale=unit_scale)
        self.amplitude = float(loading["A"]) * unit_scale
        self.period = float(loading["Ts"])
        self.tolerance = tolerance

        if validate_domain:
            self._validate_reference_domain()

        self.region = classify_mesh_regions(
            self.reference_pos,
            self.geometry,
            tolerance=tolerance,
        )
        self.motion_weight = compute_motion_weights(
            self.reference_pos,
            self.geometry,
            self.region,
        )

    def _validate_reference_domain(self) -> None:
        r = self.reference_pos[:, 0]
        z = self.reference_pos[:, 1]
        expected = np.array(
            [
                self.geometry.r1,
                self.geometry.r2,
                self.geometry.b1,
                self.geometry.fluid_top,
            ]
        )
        actual = np.array([r.min(), r.max(), z.min(), z.max()])
        if not np.allclose(actual, expected, rtol=0.0, atol=self.tolerance):
            raise ValueError(
                "Reference liquid-mesh bounds do not match JSON geometry. "
                f"Expected [Rmin, Rmax, Zmin, Zmax]={expected.tolist()}, "
                f"got {actual.tolist()}."
            )

    def at_time(self, t: float) -> MeshState:
        middle = piston_displacement(t, self.amplitude, self.period)
        middle_velocity = piston_velocity(t, self.amplitude, self.period)
        node_displacement = self.motion_weight * middle
        mesh_velocity = np.zeros_like(self.reference_pos)
        mesh_velocity[:, 1] = self.motion_weight * middle_velocity
        pos = self.reference_pos.copy()
        pos[:, 1] += node_displacement
        return MeshState(
            time=float(t),
            pos=pos,
            region=self.region.copy(),
            motion_weight=self.motion_weight.copy(),
            node_displacement=node_displacement,
            piston_displacement=middle,
            mesh_velocity=mesh_velocity,
            piston_velocity=middle_velocity,
        )
