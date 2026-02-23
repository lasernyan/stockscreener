"""
EDINET API fetcher.

Retrieves recent filings of interest:
  - 公開買付届出書   (Tender Offer)       doc_type_code = "030"
  - 大量保有報告書   (Large Shareholding)  doc_type_code = "040"
  - 自己株式取得状況報告書 (Share Repurchase) doc_type_code = "020"

EDINET v2 Document List API endpoint:
  https://disclosure.edinet-fsa.go.jp/api/v2/documents.json

Docs:
  https://disclosure.edinet-fsa.go.jp/E01EW/BLMainController.jsp?uji.verb=W1E63010CXW1E63010B&uji.bean=ee.bean.W1E63010.EEW1E63010Bean&TID=W1E63010&PID=W1E63010&SESSIONKEY=&lgKbn=2&dflg=0&iflg=0
"""
from __future__ import annotations

import logging
import os
import time
from datetime import date, timedelta
from typing import List, Optional

import requests

from stockanalyzeV2.config import EDINET_API_KEY, EDINET_DOC_TYPES
from stockanalyzeV2.models.sector_models import EDINETEvent

logger = logging.getLogger(__name__)

EDINET_BASE_URL = "https://disclosure.edinet-fsa.go.jp/api/v2"


class EDINETFetcher:
    """
    Fetch EDINET filings for the given date range.

    Parameters
    ----------
    api_key : str
        EDINET API key (Subscription Key). Falls back to env EDINET_API_KEY.
    lookback_days : int
        How many calendar days back to search.
    request_delay : float
        Seconds between API calls to respect rate limits.
    """

    def __init__(
        self,
        api_key:       str   = "",
        lookback_days: int   = 30,
        request_delay: float = 0.5,
    ):
        self.api_key       = api_key or EDINET_API_KEY or os.getenv("EDINET_API_KEY", "55f1ffb3726749b3aeba0414982e486b")
        self.lookback_days = lookback_days
        self.request_delay = request_delay
        self.session       = requests.Session()
        self.session.headers.update({"Accept": "application/json"})

    # ------------------------------------------------------------------
    def fetch_events(self, target_date: Optional[date] = None) -> List[EDINETEvent]:
        """
        Fetch all relevant filing events from EDINET for a date range.

        Parameters
        ----------
        target_date : date, optional
            End date of the search range (defaults to today).

        Returns
        -------
        List[EDINETEvent]
        """
        end_date   = target_date or date.today()
        start_date = end_date - timedelta(days=self.lookback_days)

        if not self.api_key:
            logger.warning(
                "EDINET_API_KEY not set. Skipping EDINET fetch. "
                "Set env variable EDINET_API_KEY or edit config.py."
            )
            return []

        events: List[EDINETEvent] = []
        current = start_date
        while current <= end_date:
            day_events = self._fetch_day(current)
            events.extend(day_events)
            current += timedelta(days=1)
            time.sleep(self.request_delay)

        logger.info(
            "EDINET: fetched %d events from %s to %s",
            len(events), start_date, end_date,
        )
        return events

    # ------------------------------------------------------------------
    def _fetch_day(self, target_date: date) -> List[EDINETEvent]:
        """Fetch document list for a single calendar day."""
        date_str = target_date.strftime("%Y-%m-%d")
        url      = f"{EDINET_BASE_URL}/documents.json"
        params   = {
            "date": date_str,
            "type": 2,                  # 2 = 書類一覧(メタデータ取得)
            "Subscription-Key": self.api_key,
        }

        try:
            resp = self.session.get(url, params=params, timeout=15)
            resp.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("EDINET request failed for %s: %s", date_str, exc)
            return []

        try:
            data = resp.json()
        except ValueError:
            logger.warning("EDINET: non-JSON response for %s", date_str)
            return []

        results = data.get("results", [])
        events: List[EDINETEvent] = []

        for doc in results:
            doc_type_code = str(doc.get("docTypeCode", "")).strip()
            doc_type_label = self._resolve_doc_type(doc_type_code)
            if doc_type_label is None:
                continue

            event = EDINETEvent(
                doc_type    = doc_type_label,
                filer_name  = doc.get("filerName", ""),
                filing_date = doc.get("submitDateTime", date_str)[:10],
                ticker      = doc.get("secCode", None) or None,
                description = doc.get("docDescription", ""),
                doc_id      = doc.get("docID", ""),
            )
            # Normalise ticker: EDINET uses 5-digit codes, TSE uses 4-digit + ".T"
            if event.ticker and len(event.ticker) == 5:
                event.ticker = event.ticker[:4] + ".T"

            events.append(event)

        return events

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_doc_type(code: str) -> Optional[str]:
        """Map EDINET docTypeCode → human-readable label, or None if not tracked."""
        for label, tracked_code in EDINET_DOC_TYPES.items():
            if code == tracked_code:
                return label
        return None

    # ------------------------------------------------------------------
    def summarise(self, events: List[EDINETEvent]) -> dict:
        """
        Return a summary dict of event counts by type, useful for logging.
        """
        summary: dict = {}
        for ev in events:
            summary[ev.doc_type] = summary.get(ev.doc_type, 0) + 1
        return summary
