"""
e-STAT API fetcher.

Retrieves macro-economic time series from the Japanese government statistics portal.
API reference: https://www.e-stat.go.jp/api/

Endpoints used:
  - /rest/3.0/app/json/getStatsData  → Fetch time-series data for a statsDataId

Authentication:
  appId (application ID) is required. Register for free at:
  https://www.e-stat.go.jp/api/api-info/api-guide
"""
from __future__ import annotations

import logging
import os
import time
from typing import List, Optional, Dict, Any

import requests

from stockanalyzeV2.config import ESTAT_API_KEY, ESTAT_SERIES
from stockanalyzeV2.models.sector_models import ESTATIndicator

logger = logging.getLogger(__name__)

ESTAT_BASE_URL = "https://api.e-stat.go.jp/rest/3.0/app/json"


class ESTATFetcher:
    """
    Fetch macro indicators from the e-STAT API.

    Parameters
    ----------
    app_id : str
        e-STAT application ID. Falls back to env ESTAT_APP_ID.
    request_delay : float
        Seconds between API calls.
    """

    def __init__(self, app_id: str = "", request_delay: float = 0.5):
        self.app_id        = app_id or ESTAT_API_KEY or os.getenv("ESTAT_APP_ID", "")
        self.request_delay = request_delay
        self.session       = requests.Session()

    # ------------------------------------------------------------------
    def fetch_all(self, series: Optional[Dict[str, str]] = None) -> List[ESTATIndicator]:
        """
        Fetch all indicator series defined in config.ESTAT_SERIES (or override).

        Parameters
        ----------
        series : dict, optional
            {series_name: statsDataId} mapping. Defaults to config.ESTAT_SERIES.

        Returns
        -------
        List[ESTATIndicator]
        """
        if not self.app_id:
            logger.warning(
                "ESTAT_APP_ID not set. Skipping e-STAT fetch. "
                "Set env variable ESTAT_APP_ID or edit config.py."
            )
            return []

        series = series or ESTAT_SERIES
        indicators: List[ESTATIndicator] = []

        for name, stats_data_id in series.items():
            indicator = self._fetch_series(name, stats_data_id)
            if indicator:
                indicators.append(indicator)
            time.sleep(self.request_delay)

        logger.info("e-STAT: fetched %d indicators", len(indicators))
        return indicators

    # ------------------------------------------------------------------
    def _fetch_series(self, name: str, stats_data_id: str) -> Optional[ESTATIndicator]:
        """Fetch a single time series and return its latest/previous values."""
        url    = f"{ESTAT_BASE_URL}/getStatsData"
        params = {
            "appId":         self.app_id,
            "statsDataId":   stats_data_id,
            "metaGetFlg":    "N",          # skip metadata for speed
            "cntGetFlg":     "N",
            "sectionHeaderFlg": "1",
            "lang":          "J",
            "limit":         5,            # last 5 observations
        }

        try:
            resp = self.session.get(url, params=params, timeout=20)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("e-STAT request failed for %s (%s): %s", name, stats_data_id, exc)
            return None

        try:
            body = resp.json()
        except ValueError:
            logger.warning("e-STAT: non-JSON response for %s", name)
            return None

        return self._parse_response(name, stats_data_id, body)

    # ------------------------------------------------------------------
    @staticmethod
    def _parse_response(
        name:          str,
        stats_data_id: str,
        body:          Dict[str, Any],
    ) -> Optional[ESTATIndicator]:
        """Extract the most recent value from a getStatsData response."""
        try:
            statistical_data = body["GET_STATS_DATA"]["STATISTICAL_DATA"]
            data_inf         = statistical_data["DATA_INF"]
            value_list       = data_inf["VALUE"]
        except (KeyError, TypeError) as exc:
            logger.debug("e-STAT parse error for %s: %s", name, exc)
            return None

        if not value_list:
            return None

        # The list is already limited to 5; take the two most recent entries.
        # e-STAT returns values as strings; "$" key holds the numeric value.
        def _safe_float(v: Any) -> Optional[float]:
            try:
                return float(str(v).replace(",", "").strip())
            except (ValueError, TypeError):
                return None

        latest_entry = value_list[-1]
        prev_entry   = value_list[-2] if len(value_list) >= 2 else None

        latest_val = _safe_float(latest_entry.get("$"))
        prev_val   = _safe_float(prev_entry.get("$")) if prev_entry else None
        change     = (latest_val - prev_val) if (latest_val is not None and prev_val is not None) else None

        # Reference date is typically encoded in "@time" attribute (e.g. "2024000100")
        ref_date = str(latest_entry.get("@time", ""))

        return ESTATIndicator(
            series_name    = name,
            series_id      = stats_data_id,
            latest_value   = latest_val,
            prev_value     = prev_val,
            change         = change,
            unit           = "",        # unit metadata skipped for speed
            reference_date = ref_date,
        )
