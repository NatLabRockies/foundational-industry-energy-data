"""Census Bureau's Quarterly Survey of Plant Capacity Utilization

Methods for downloading and formatting operating hours reported under the Census Bureau's Quarterly Survey of Plant Capacity Utilization.
"""

from .census_qpc import weekly_operating_hours

__all__ = ["weekly_operating_hours"]
