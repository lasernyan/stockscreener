"""
Rotation Detector.

Detects inter-sector capital rotation by comparing two sets of SectorMetrics
(current snapshot vs. a previous snapshot).

Detection rules:
  - Rising Sector:  SCS change ≥ +RISING_THRESHOLD  (default +3)
  - Falling Sector: SCS change ≤ -FALLING_THRESHOLD (default -3)
  - Rotation Pair:  Any (Falling → Rising) combination in the same market.

The detector also annotates each SectorMetrics object in-place with:
  - rotation_status  (RISING / FALLING / STABLE)
  - scs_change       (numeric delta vs. previous snapshot)
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional

from stockanalyzeV2.config import RotationConfig
from stockanalyzeV2.models.sector_models import (
    SectorMetrics,
    RotationPair,
    RotationStatus,
)

logger = logging.getLogger(__name__)


class RotationDetector:
    """
    Compare two snapshots of sector metrics to detect capital rotation.

    Parameters
    ----------
    config : RotationConfig
        Rising/falling thresholds and bonus score configuration.
    """

    def __init__(self, config: RotationConfig = None):
        self.config = config or RotationConfig()

    # ------------------------------------------------------------------
    def detect(
        self,
        current:  List[SectorMetrics],
        previous: Optional[List[SectorMetrics]] = None,
    ) -> List[RotationPair]:
        """
        Detect rotation pairs between current and previous snapshots.

        If *previous* is None or empty the method still annotates each
        SectorMetrics with rotation_status = STABLE and scs_change = 0.

        Parameters
        ----------
        current : List[SectorMetrics]
            Most-recent sector metrics.
        previous : List[SectorMetrics], optional
            Prior-period sector metrics for comparison.

        Returns
        -------
        List[RotationPair]
            Detected rotation pairs (may be empty).
        """
        if not previous:
            logger.info("No previous snapshot available – rotation detection skipped.")
            for sm in current:
                sm.rotation_status = RotationStatus.STABLE
                sm.scs_change      = 0.0
            return []

        # Build lookup: sector → previous SCS
        prev_map: Dict[str, float] = {
            sm.sector: sm.scs for sm in previous
        }

        rising:  List[SectorMetrics] = []
        falling: List[SectorMetrics] = []

        for sm in current:
            prev_scs    = prev_map.get(sm.sector)
            if prev_scs is None:
                sm.scs_change      = 0.0
                sm.rotation_status = RotationStatus.STABLE
                continue

            delta = sm.scs - prev_scs
            sm.scs_change = delta

            if delta >= self.config.rising_threshold:
                sm.rotation_status = RotationStatus.RISING
                rising.append(sm)
                logger.debug(
                    "Rising sector: %s  SCS %+.1f → %+.1f (Δ%+.1f)",
                    sm.sector, prev_scs, sm.scs, delta,
                )
            elif delta <= self.config.falling_threshold:
                sm.rotation_status = RotationStatus.FALLING
                falling.append(sm)
                logger.debug(
                    "Falling sector: %s  SCS %+.1f → %+.1f (Δ%+.1f)",
                    sm.sector, prev_scs, sm.scs, delta,
                )
            else:
                sm.rotation_status = RotationStatus.STABLE

        # Build rotation pairs: every (falling, rising) combination
        pairs: List[RotationPair] = []
        for fall in falling:
            for rise in rising:
                if fall.market != rise.market:
                    continue   # only intra-market rotations
                pair = RotationPair(
                    from_sector      = fall.sector,
                    to_sector        = rise.sector,
                    from_scs_change  = fall.scs_change,
                    to_scs_change    = rise.scs_change,
                    market           = fall.market,
                )
                pairs.append(pair)
                logger.info(
                    "Rotation pair detected: %s → %s  (%s) [Δ%.1f / Δ%.1f]",
                    fall.sector, rise.sector, fall.market,
                    fall.scs_change, rise.scs_change,
                )

        return pairs

    # ------------------------------------------------------------------
    def rising_sectors(self, metrics: List[SectorMetrics]) -> List[SectorMetrics]:
        """Filter for sectors already annotated as RISING."""
        return [sm for sm in metrics if sm.rotation_status == RotationStatus.RISING]

    def falling_sectors(self, metrics: List[SectorMetrics]) -> List[SectorMetrics]:
        """Filter for sectors already annotated as FALLING."""
        return [sm for sm in metrics if sm.rotation_status == RotationStatus.FALLING]
