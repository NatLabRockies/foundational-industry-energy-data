import logging
from pathlib import Path

import polars as pl
import pooch
from pooch import HTTPDownloader


module_logger = logging.getLogger(__name__)


def fetch_scc():
    """Load EPA's Source Classification Codes (SCC)

    Note that downloading directly from website assignes filename for
    csv based as 'SCCDownload-{y}-{md}-{t}.csv'

    Should force the filename to be 'SCCDownload.csv'?

    This was originally defined in scc/scc_unit_id.py.

    payload = {
      "format": "CSV",
      "sortFacet": "scc level one",
      "filename": "SCCDownload.csv",
      }

    This is a slow service, thus require a long timeout.
    """
    module_logger.debug("Fetching SCC data")

    fname = pooch.retrieve(
        url="https://sor-scc-api.epa.gov/sccwebservices/v1/SCC?format=CSV&sortFacet=scc+level+one&filename=SCCDownload.csv",
        # Change too often. This was the last version I downloaded
        # known_hash="sha256:51a71b727adf21c40617d80f8def6d31c58ab13676a2c06ab73f31a8b0828ac6",
        known_hash=None,
        path=pooch.os_cache("FIED/SCC"),
        downloader=HTTPDownloader(progressbar=True, timeout=300, verify=False),
    )
    module_logger.debug(f"SCC raw dataset ready: {fname}")

    lf = pl.scan_csv(
        fname,
        dtypes={
            "SCC": pl.String,
            "last edited date": pl.String,
        },
    )

    # Drop last column, which is empty (all values are null)
    lf = lf.drop("")

    return lf.rename(
        {col: col.replace(" ", "_") for col in lf.collect_schema().names()}
    )
