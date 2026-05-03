"""UAV rotary-wing energy model for ISAC trajectory planning.

Implements the propulsion power model of Zeng, Xu & Zhang (2019),
as adopted in equation (10) of Zhou et al. 2025.
"""

from typing import Sequence, Tuple

import numpy as np


class UAVEnergyModel:
    """Rotary-wing UAV energy model with optional ISAC payload accounting.

    Default coefficients follow the rotary-wing parameters from
    Zeng-Xu-Zhang 2019, also cited in equation (10) of Zhou et al. 2025.
    """

    def __init__(
        self,
        P0: float = 79.86,
        P1: float = 88.63,
        U_tip: float = 120.0,
        v0: float = 4.03,
        d_f_rho_s_A: float = 0.6,
    ) -> None:
        self.P0: float = P0
        self.P1: float = P1
        self.U_tip: float = U_tip
        self.v0: float = v0
        self.d_f_rho_s_A: float = d_f_rho_s_A

    def flight_power(self, v: float) -> float:
        """Propulsion power [W] at forward speed v [m/s] (eq. 10, Zhou et al. 2025)."""
        blade_profile = self.P0 * (1.0 + 3.0 * v ** 2 / self.U_tip ** 2)
        induced = self.P1 * np.sqrt(
            np.sqrt(1.0 + v ** 4 / (4.0 * self.v0 ** 4)) - v ** 2 / (2.0 * self.v0 ** 2)
        )
        parasite = 0.5 * self.d_f_rho_s_A * v ** 3
        return float(blade_profile + induced + parasite)

    def flight_energy(self, distance: float, v: float) -> float:
        """Energy [J] to traverse `distance` [m] at constant speed `v` [m/s]."""
        return self.flight_power(v) * distance / v

    def isac_energy(self, P_sar: float, P_com: float, duration: float) -> float:
        """Energy [J] consumed by SAR + communication payloads over `duration` [s]."""
        return (P_sar + P_com) * duration

    def total_path_energy(
        self,
        points: Sequence[Tuple[float, float]],
        path: Sequence[int],
        v_cruise: float = 20.0,
        P_sar: float = 2.5,
        P_com: float = 1.0,
        dwell_time: float = 2.0,
    ) -> float:
        """Sum per-edge flight energy and per-node ISAC sensing energy along `path`."""
        pts = np.asarray(points, dtype=float)
        idx = np.asarray(path, dtype=int)

        edges = pts[idx[1:]] - pts[idx[:-1]]
        edge_lengths = np.linalg.norm(edges, axis=1)
        flight_e = float(np.sum([self.flight_energy(d, v_cruise) for d in edge_lengths]))

        sensing_e = self.isac_energy(P_sar, P_com, dwell_time) * len(idx)

        return flight_e + sensing_e


if __name__ == "__main__":
    model = UAVEnergyModel()
    for v in [5, 10, 15, 20, 25, 30]:
        print(f"v = {v:>3} m/s   P(v) = {model.flight_power(float(v)):8.2f} W")
