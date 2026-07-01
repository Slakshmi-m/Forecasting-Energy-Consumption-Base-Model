"""Typed exception hierarchy for energy_forecast.

All modules in this package raise these exceptions rather than
propagating raw third-party exceptions to callers.
"""


class ENTSOEFetchError(RuntimeError):
    """Raised when an ENTSO-E API call fails or returns empty data."""


class WeatherFetchError(RuntimeError):
    """Raised when ERA5 weather data fetch fails or returns incomplete data."""


class PreprocessingError(ValueError):
    """Raised when input data fails validation during preprocessing."""


class ModelComplianceError(RuntimeError):
    """Raised when a model violates spotforecast2-safe API contracts."""
