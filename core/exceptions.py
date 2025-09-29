"""Custom exceptions for the SOAP evaluation system."""


class DataLoadingError(Exception):
    """Raised when data loading fails"""
    pass


class FieldDetectionError(Exception):
    """Raised when field detection fails"""
    pass


class SOAPGenerationError(Exception):
    """Raised when SOAP generation fails"""
    pass


class EvaluationError(Exception):
    """Raised when evaluation fails"""
    pass
