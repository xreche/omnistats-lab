"""Custom exceptions for OmniStats Lab."""


class DataValidationError(Exception):
    """Raised when data validation fails."""
    pass


class ModelTrainingError(Exception):
    """Raised when model training fails."""
    pass


class DataLoadingError(Exception):
    """Raised when data loading fails."""
    pass

