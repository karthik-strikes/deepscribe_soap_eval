"""
Universal Data Loading and Field Detection System
=================================================

Automatically loads medical datasets from multiple sources (HuggingFace, CSV, JSON)
and uses DSPy to intelligently detect field types for SOAP note evaluation.
"""

import json
import pandas as pd
import asyncio
import logging
from datasets import load_dataset
from typing import Dict, List, Any, Union, Optional, Tuple
import dspy
from dataclasses import dataclass
from enum import Enum
import os
import re
from core.exceptions import DataLoadingError, FieldDetectionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FieldMapping:
    """
    Container for field detection results.

    Stores which fields contain transcripts, reference notes, ground truth, and patient metadata,
    along with confidence scores and reasoning.
    """
    transcript_field: str
    reference_notes_field: str
    ground_truth_field: str
    patient_metadata_fields: List[str]
    confidence_score: float
    reasoning: str


class DataSourceType(Enum):
    """
    Supported data source types.

    HUGGINGFACE: HuggingFace datasets hub
    CSV: Local CSV files
    JSON: Local JSON files
    UNKNOWN: Unrecognized source type
    """
    HUGGINGFACE = "huggingface"
    CSV = "csv"
    JSON = "json"
    UNKNOWN = "unknown"


class FieldDetectionSignature(dspy.Signature):
    """Analyze medical dataset fields to identify their purpose for medical note evaluation."""

    field_names: str = dspy.InputField(
        desc="Comma-separated list of all field names in the dataset")
    field_samples: str = dspy.InputField(
        desc="Sample content from each field (field_name: sample_content)")

    transcript_field: str = dspy.OutputField(
        desc="Field name containing patient-provider conversation/dialogue (empty string if none found)")
    reference_notes_field: str = dspy.OutputField(
        desc="Field name containing SOAP notes or medical notes that can be used as reference for evaluation (includes doctor-written notes) (empty string if none found)")
    ground_truth_field: str = dspy.OutputField(
        desc="Field name containing SOAP notes specifically written by doctors/physicians (can be same as reference_notes_field if all references are doctor-written) (empty string if none found)")
    patient_metadata_fields: str = dspy.OutputField(
        desc="Comma-separated list of field names containing demographic/medical metadata (age, gender, vitals, etc.)")
    confidence_score: float = dspy.OutputField(
        desc="Confidence level between 0-1 for field identification accuracy")
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how fields were identified")


class DSPyFieldDetector:
    """
    Intelligent field type detector using DSPy LLM reasoning.

    Analyzes dataset structure and sample content to automatically identify
    which fields contain transcripts, reference notes, and patient metadata.
    Falls back to keyword-based detection if LLM fails.
    """

    def __init__(self):
        """Initialize the DSPy field detector."""
        self.field_detector = dspy.Predict(FieldDetectionSignature)

    async def detect_fields(self, sample_data: Dict[str, Any]) -> FieldMapping:
        """
        Use DSPy LLM to analyze sample data and detect field purposes.

        Args:
            sample_data: Dictionary containing one sample row from the dataset

        Returns:
            FieldMapping with detected field names and confidence score

        Raises:
            FieldDetectionError: If detection fails completely
        """
        try:
            field_names = ", ".join(sample_data.keys())

            # Create sample content for each field (truncated to 150 chars)
            field_samples = []
            for field_name, value in sample_data.items():
                sample_content = str(value)[:150] if value else "empty"
                field_samples.append(f"{field_name}: {sample_content}...")

            field_samples_str = "\n".join(field_samples)

            # Call DSPy
            try:
                result = await asyncio.to_thread(
                    self.field_detector,
                    field_names=field_names,
                    field_samples=field_samples_str
                )
            except Exception as e:
                logger.warning(
                    f"DSPy field detection failed: {e}, using fallback")
                return self._fallback_detection(sample_data)

            # Parse metadata fields
            metadata_fields = [
                f.strip() for f in result.patient_metadata_fields.split(",") if f.strip()
            ]

            field_mapping = FieldMapping(
                transcript_field=result.transcript_field.strip(),
                reference_notes_field=result.reference_notes_field.strip(),
                ground_truth_field=result.ground_truth_field.strip(
                ) if hasattr(result, 'ground_truth_field') else "",
                patient_metadata_fields=metadata_fields,
                confidence_score=float(result.confidence_score),
                reasoning=result.reasoning
            )

            # Validate that at least transcript field was found
            if not field_mapping.transcript_field:
                logger.warning(
                    "DSPy did not detect transcript field, using fallback")
                return self._fallback_detection(sample_data)

            logger.info(f"Field detection successful: transcript='{field_mapping.transcript_field}', "
                        f"reference='{field_mapping.reference_notes_field}', "
                        f"confidence={field_mapping.confidence_score:.2f}")

            return field_mapping

        except Exception as e:
            logger.error(f"DSPy field detection failed: {e}")
            raise FieldDetectionError(f"DSPy field detection failed: {e}")

    def _fallback_detection(self, sample_data: Dict[str, Any]) -> FieldMapping:
        """
        Enhanced keyword-based fallback detection when DSPy fails.

        Uses predefined keyword lists to match field names with their likely purposes.

        Args:
            sample_data: Dictionary containing one sample row from the dataset

        Returns:
            FieldMapping with detected fields and confidence score
        """
        logger.info("Using enhanced keyword-based fallback detection")

        # Keyword candidates for different field types
        transcript_candidates = ['transcript', 'dialogue',
                                 'conversation', 'patient_convo', 'text', 'input']
        reference_notes_candidates = ['soap_note', 'soap', 'note', 'summary', 'reference_notes',
                                      'existing_notes', 'clinical_note', 'output']
        metadata_candidates = ['age', 'gender', 'patient_name', 'dob', 'phone', 'address',
                               'vitals', 'demographics', 'patient_id']

        def find_best_match(candidates: List[str], fields: List[str]) -> str:
            """Find first field name containing any candidate keyword."""
            for field in fields:
                field_lower = field.lower()
                for candidate in candidates:
                    if candidate in field_lower:
                        logger.debug(
                            f"Matched field '{field}' with candidate '{candidate}'")
                        return field
            return ""

        fields = list(sample_data.keys())
        transcript_field = find_best_match(transcript_candidates, fields)
        reference_notes_field = find_best_match(
            reference_notes_candidates, fields)

        # Find metadata fields
        metadata_fields = []
        for field in fields:
            if field not in [transcript_field, reference_notes_field]:
                for meta_candidate in metadata_candidates:
                    if meta_candidate.lower() in field.lower():
                        metadata_fields.append(field)
                        break

        # Calculate confidence based on matches found
        confidence = 0.0
        if transcript_field:
            confidence += 0.4
        if reference_notes_field:
            confidence += 0.3
        if metadata_fields:
            confidence += min(0.3, len(metadata_fields) * 0.1)

        logger.info(f"Fallback detection: transcript='{transcript_field}', "
                    f"reference='{reference_notes_field}', confidence={confidence:.2f}")

        return FieldMapping(
            transcript_field=transcript_field,
            reference_notes_field=reference_notes_field,
            ground_truth_field="",  # Fallback doesn't detect ground truth
            patient_metadata_fields=metadata_fields,
            confidence_score=confidence,
            reasoning="Keyword-based fallback detection"
        )


class UniversalDataLoader:
    """
    Universal data loader supporting multiple source types.

    Automatically detects source type, loads data, and normalizes field names
    using intelligent field detection. Supports HuggingFace datasets, CSV, and JSON files.
    """

    def __init__(self, field_detector: DSPyFieldDetector):
        """
        Initialize data loader with field detector.

        Args:
            field_detector: DSPyFieldDetector instance for intelligent field mapping
        """
        self.field_detector = field_detector
        self._cached_mappings = {}

    async def load_and_normalize(self, source: str, source_type: Optional[str] = None,
                                 max_samples: int = 100) -> Tuple[List[Dict[str, Any]], FieldMapping]:
        """
        Load data from any source and normalize using DSPy field detection.

        Args:
            source: Data source (HuggingFace dataset name, file path, etc.)
            source_type: Optional explicit source type ("huggingface", "csv", "json")
            max_samples: Maximum number of samples to load

        Returns:
            Tuple of (normalized_data_list, field_mapping)

        Raises:
            DataLoadingError: If loading or normalization fails
        """
        # Validate max_samples
        if max_samples <= 0:
            raise DataLoadingError("max_samples must be greater than 0")
        if max_samples > 100000:
            logger.warning(
                f"max_samples={max_samples} is very large, consider reducing for performance")

        try:
            logger.info(
                f"Loading data from {source} (max_samples={max_samples})")

            # Load raw data
            raw_data = self._load_raw_data(source, source_type, max_samples)
            if not raw_data:
                raise DataLoadingError(f"No data loaded from {source}")

            logger.info(f"Loaded {len(raw_data)} raw records")

            # Get sample for field detection
            sample_data = raw_data[0]

            # Check cache or perform field detection
            cache_key = self._create_cache_key(sample_data)
            if cache_key in self._cached_mappings:
                logger.info("Using cached field mapping")
                field_mapping = self._cached_mappings[cache_key]
            else:
                logger.info("Performing field detection")
                try:
                    field_mapping = await self.field_detector.detect_fields(sample_data)
                except FieldDetectionError:
                    logger.warning("Field detection failed, using fallback")
                    field_mapping = self.field_detector._fallback_detection(
                        sample_data)

                # Validate field mapping has at least transcript
                if not field_mapping.transcript_field:
                    raise DataLoadingError(
                        "Could not detect transcript field in dataset. "
                        "Please ensure dataset contains conversation/dialogue data."
                    )

                self._cached_mappings[cache_key] = field_mapping

            # Normalize all rows
            normalized_data = []
            failed_rows = []

            for i, row in enumerate(raw_data):
                normalized_row = self._normalize_row(
                    row, field_mapping, row_index=i)
                if normalized_row:
                    normalized_data.append(normalized_row)
                else:
                    failed_rows.append(i)

            # Log normalization results
            if failed_rows:
                logger.warning(
                    f"Failed to normalize {len(failed_rows)} rows (indices: {failed_rows[:10]}...)")

            logger.info(
                f"Successfully normalized {len(normalized_data)}/{len(raw_data)} records")

            if not normalized_data:
                raise DataLoadingError(
                    f"No valid data after normalization from {source}. "
                    "Check that transcript field contains non-empty values."
                )

            return normalized_data, field_mapping

        except Exception as e:
            logger.error(
                f"Failed to load and normalize data from {source}: {e}")
            raise DataLoadingError(
                f"Failed to load and normalize data from {source}: {e}")

    def _load_raw_data(self, source: str, source_type: Optional[str], max_samples: int) -> List[Dict]:
        """
        Load raw data from source based on detected or specified type.

        Args:
            source: Data source identifier
            source_type: Optional explicit source type
            max_samples: Maximum records to load

        Returns:
            List of dictionaries containing raw data
        """
        if source_type is None:
            source_type = self._detect_source_type(source)
            logger.info(f"Auto-detected source type: {source_type}")

        if source_type == DataSourceType.HUGGINGFACE.value:
            return self._load_huggingface(source, max_samples)
        elif source_type == DataSourceType.CSV.value:
            return self._load_csv(source, max_samples)
        elif source_type == DataSourceType.JSON.value:
            return self._load_json(source, max_samples)
        else:
            raise DataLoadingError(f"Unsupported source type: {source_type}")

    def _load_huggingface(self, dataset_name: str, max_samples: int) -> List[Dict]:
        """
        Load data from HuggingFace datasets hub.

        Args:
            dataset_name: HuggingFace dataset identifier (e.g., "username/dataset")
            max_samples: Maximum records to load

        Returns:
            List of dictionaries

        Raises:
            DataLoadingError: If loading fails
        """
        try:
            # Validate HuggingFace dataset name pattern
            if not self._is_valid_hf_dataset(dataset_name):
                raise DataLoadingError(
                    f"Invalid HuggingFace dataset pattern: {dataset_name}. "
                    "Expected format: 'username/dataset-name'"
                )

            logger.info(f"Loading HuggingFace dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")
            records = [dict(row) for row in dataset.select(
                range(min(len(dataset), max_samples)))]
            logger.info(f"Loaded {len(records)} records from HuggingFace")
            return records

        except Exception as e:
            logger.error(
                f"Error loading HuggingFace dataset {dataset_name}: {e}")
            raise DataLoadingError(
                f"Error loading HuggingFace dataset {dataset_name}: {e}")

    def _load_csv(self, file_path: str, max_samples: int) -> List[Dict]:
        """
        Load data from CSV file.

        Args:
            file_path: Path to CSV file
            max_samples: Maximum records to load

        Returns:
            List of dictionaries

        Raises:
            DataLoadingError: If file not found or loading fails
        """
        try:
            if not os.path.exists(file_path):
                raise DataLoadingError(f"CSV file not found: {file_path}")

            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, nrows=max_samples)
            records = df.to_dict('records')
            logger.info(f"Loaded {len(records)} records from CSV")
            return records

        except Exception as e:
            logger.error(f"Error loading CSV {file_path}: {e}")
            raise DataLoadingError(f"Error loading CSV {file_path}: {e}")

    def _load_json(self, file_path: str, max_samples: int) -> List[Dict]:
        """
        Load data from JSON file.

        Args:
            file_path: Path to JSON file
            max_samples: Maximum records to load

        Returns:
            List of dictionaries

        Raises:
            DataLoadingError: If file not found or loading fails
        """
        try:
            if not os.path.exists(file_path):
                raise DataLoadingError(f"JSON file not found: {file_path}")

            logger.info(f"Loading JSON file: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)

            records = data[:max_samples] if isinstance(data, list) else [data]
            logger.info(f"Loaded {len(records)} records from JSON")
            return records

        except Exception as e:
            logger.error(f"Error loading JSON {file_path}: {e}")
            raise DataLoadingError(f"Error loading JSON {file_path}: {e}")

    def _detect_source_type(self, source: str) -> str:
        """
        Detect source type from source identifier.

        Args:
            source: Source identifier (file path or dataset name)

        Returns:
            Source type string
        """
        if source.endswith('.csv'):
            return DataSourceType.CSV.value
        elif source.endswith('.json'):
            return DataSourceType.JSON.value
        elif self._is_valid_hf_dataset(source):
            return DataSourceType.HUGGINGFACE.value
        else:
            return DataSourceType.UNKNOWN.value

    def _is_valid_hf_dataset(self, source: str) -> bool:
        """
        Check if source matches HuggingFace dataset pattern.

        Valid patterns: username/dataset-name, org/dataset.name, etc.

        Args:
            source: Source identifier to check

        Returns:
            True if valid HuggingFace dataset pattern
        """
        # Pattern: org_name/dataset_name (allows alphanumeric, hyphens, underscores, dots)
        pattern = r'^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$'
        return bool(re.match(pattern, source)) and not source.startswith('http')

    def _create_cache_key(self, sample_data: Dict) -> str:
        """
        Create cache key for field mapping based on dataset structure.

        Uses sorted field names to create consistent cache keys.

        Args:
            sample_data: Sample data dictionary

        Returns:
            Cache key string
        """
        field_names = sorted(sample_data.keys())
        return "|".join(field_names)

    def _normalize_row(self, row: Dict, field_mapping: FieldMapping, row_index: int = -1) -> Optional[Dict[str, Any]]:
        """
        Normalize a single row to standardized format.

        Args:
            row: Raw data row
            field_mapping: Field mapping to use for normalization
            row_index: Row index for error logging

        Returns:
            Normalized dictionary or None if normalization fails
        """
        try:
            # Extract transcript (required)
            transcript = str(
                row.get(field_mapping.transcript_field, "")).strip()
            if not transcript:
                logger.debug(f"Row {row_index}: Empty transcript, skipping")
                return None

            # Extract reference notes (optional)
            reference_notes = str(
                row.get(field_mapping.reference_notes_field, "")).strip()

            # Extract ground truth (optional)
            ground_truth = str(
                row.get(field_mapping.ground_truth_field, "")).strip()

            # Extract patient metadata
            patient_metadata = {}
            for field in field_mapping.patient_metadata_fields:
                if field in row and row[field] is not None:
                    patient_metadata[field] = row[field]

            normalized_row = {
                'transcript': transcript,
                'reference_notes': reference_notes,
                'ground_truth': ground_truth,
                'patient_metadata': patient_metadata,
                'source': "auto_detected_mapping",
                'field_mapping_confidence': field_mapping.confidence_score
            }

            return normalized_row

        except Exception as e:
            logger.warning(f"Error normalizing row {row_index}: {e}")
            return None
