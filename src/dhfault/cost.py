"""Cost of a fault in CHF from excess return temperature x flow (parameterised, illustrative).

Utilities with return-temperature incentives charge (or credit) customers by how much their return
water exceeds a reference. Such "motivation tariffs" are documented for Danish district heating, see
the README references. This tool does NOT know any real utility's tariff: ``TariffConfig`` holds an
illustrative penalty that you should replace with your own.

Default: 1 K of excess return temperature on 1 m3 of water carries 1.163 kWh of heat that the plant
cannot use for generation efficiency. At the average Swiss district-heating price of about
0.17 CHF/kWh (Swiss Price Surveillance, 2024, excl. VAT) that heat is worth 0.198 CHF per K*m3. The
default penalty charges a quarter of that: 0.05 CHF per K*m3.
"""

from __future__ import annotations

from dataclasses import dataclass

C_WATER_KWH_M3K = 1.163


@dataclass(frozen=True)
class TariffConfig:
    chf_per_km3: float = 0.05  # CHF per (K of excess return temperature x m3 of flow)
    energy_chf_per_kwh: float = 0.17  # for metering-error exposure (sensor faults)

    def __post_init__(self) -> None:
        if self.chf_per_km3 < 0 or self.energy_chf_per_kwh < 0:
            raise ValueError("tariff values must be non-negative")


def retail_value_per_km3(tariff: TariffConfig) -> float:
    """CHF value of the heat carried by 1 K x 1 m3 at the retail energy price."""
    return C_WATER_KWH_M3K * tariff.energy_chf_per_kwh


def excess_cost_chf(excess_km3: float, tariff: TariffConfig | None = None) -> float:
    """Return-temperature penalty in CHF for ``excess_km3`` (K x m3). Negative values count as 0."""
    tariff = tariff or TariffConfig()
    return max(float(excess_km3), 0.0) * tariff.chf_per_km3


def metering_exposure_chf(missing_kwh: float, tariff: TariffConfig | None = None) -> float:
    """CHF of heat delivered but not metered (sensor faults). Negative values count as 0."""
    tariff = tariff or TariffConfig()
    return max(float(missing_kwh), 0.0) * tariff.energy_chf_per_kwh
