"""Universal data loading and field detection system."""

import json
import pandas as pd
from datasets import load_dataset
from typing import Dict, List, Any, Union, Optional, Tuple
import dspy
from dataclasses import dataclass
from enum import Enum
import os
import re
from core.exceptions import DataLoadingError, FieldDetectionError


@dataclass
class FieldMapping:
    """Represents the LLM's field detection results"""
    transcript_field: str
    reference_notes_field: str
    patient_metadata_fields: List[str]
    confidence_score: float
    reasoning: str


class DataSourceType(Enum):
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
        desc="Field name containing reference SOAP notes/medical summaries/existing notes (empty string if none found)")
    patient_metadata_fields: str = dspy.OutputField(
        desc="Comma-separated list of field names containing demographic/medical metadata (age, gender, vitals, etc.)")
    confidence_score: float = dspy.OutputField(
        desc="Confidence level between 0-1 for field identification accuracy")
    reasoning: str = dspy.OutputField(
        desc="Brief explanation of how fields were identified")


class DSPyFieldDetector:
    """Uses DSPy to intelligently detect field types in medical datasets"""

    def __init__(self):
        self.field_detector = dspy.Predict(FieldDetectionSignature)

    def detect_fields(self, sample_data: Dict[str, Any]) -> FieldMapping:
        """Use DSPy LLM to analyze sample data and detect field purposes"""

        try:
            field_names = ", ".join(sample_data.keys())

            field_samples = []
            for field_name, value in sample_data.items():
                sample_content = str(value)[:150] if value else "empty"
                field_samples.append(f"{field_name}: {sample_content}...")

            field_samples_str = "\n".join(field_samples)

            result = self.field_detector(
                field_names=field_names,
                field_samples=field_samples_str
            )

            metadata_fields = [
                f.strip() for f in result.patient_metadata_fields.split(",") if f.strip()]

            return FieldMapping(
                transcript_field=result.transcript_field,
                reference_notes_field=result.reference_notes_field,
                patient_metadata_fields=metadata_fields,
                confidence_score=float(result.confidence_score),
                reasoning=result.reasoning
            )

        except Exception as e:
            raise FieldDetectionError(f"DSPy field detection failed: {e}")

    def _fallback_detection(self, sample_data: Dict[str, Any]) -> FieldMapping:
        """Enhanced fallback detection when DSPy fails"""
        print("Using enhanced keyword-based fallback detection...")

        transcript_candidates = ['transcript',
                                 'dialogue', 'conversation', 'patient_convo']
        reference_notes_candidates = [
            'soap_note', 'soap', 'note', 'summary', 'reference_notes', 'existing_notes']
        metadata_candidates = ['age', 'gender',
                               'patient_name', 'dob', 'phone', 'address']

        def find_best_match(candidates, fields):
            matches = []
            for field in fields:
                field_lower = field.lower()
                for candidate in candidates:
                    if candidate in field_lower:
                        matches.append(field)
                        break
            return matches[0] if matches else ""

        fields = list(sample_data.keys())
        transcript_field = find_best_match(transcript_candidates, fields)
        reference_notes_field = find_best_match(
            reference_notes_candidates, fields)

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

        return FieldMapping(
            transcript_field=transcript_field,
            reference_notes_field=reference_notes_field,
            patient_metadata_fields=metadata_fields,
            confidence_score=confidence,
            reasoning="Keyword-based fallback detection"
        )


class UniversalDataLoader:
    """Loads data from any source and normalizes it using DSPy field detection"""

    def __init__(self, field_detector: DSPyFieldDetector):
        self.field_detector = field_detector
        self._cached_mappings = {}

    def load_and_normalize(self, source: str, source_type: Optional[str] = None,
                           max_samples: int = 100) -> Tuple[List[Dict[str, Any]], FieldMapping]:
        """Load data from any source and normalize using DSPy field detection"""

        try:
            raw_data = self._load_raw_data(source, source_type, max_samples)
            if not raw_data:
                raise DataLoadingError(f"No data loaded from {source}")

            sample_data = raw_data[0]

            cache_key = self._create_cache_key(sample_data)
            if cache_key in self._cached_mappings:
                field_mapping = self._cached_mappings[cache_key]
                print(f"Using cached field mapping for {source}")
            else:
                print(f"Analyzing fields for {source} using DSPy...")
                try:
                    field_mapping = self.field_detector.detect_fields(
                        sample_data)
                except FieldDetectionError:
                    print("DSPy detection failed, using fallback...")
                    field_mapping = self.field_detector._fallback_detection(
                        sample_data)

                self._cached_mappings[cache_key] = field_mapping

            normalized_data = []
            for row in raw_data:
                normalized_row = self._normalize_row(row, field_mapping)
                if normalized_row:
                    normalized_data.append(normalized_row)

            if not normalized_data:
                raise DataLoadingError(
                    f"No valid data after normalization from {source}")

            print(
                f"Successfully normalized {len(normalized_data)} samples from {source}")
            return normalized_data, field_mapping

        except Exception as e:
            raise DataLoadingError(
                f"Failed to load and normalize data from {source}: {e}")

    def _load_raw_data(self, source: str, source_type: Optional[str], max_samples: int) -> List[Dict]:
        if source_type is None:
            source_type = self._detect_source_type(source)

        if source_type == DataSourceType.HUGGINGFACE.value:
            return self._load_huggingface(source, max_samples)
        elif source_type == DataSourceType.CSV.value:
            return self._load_csv(source, max_samples)
        elif source_type == DataSourceType.JSON.value:
            return self._load_json(source, max_samples)
        else:
            raise DataLoadingError(f"Unsupported source type: {source_type}")

    def _load_huggingface(self, dataset_name: str, max_samples: int) -> List[Dict]:
        try:
            # Validate HuggingFace dataset name pattern
            if not self._is_valid_hf_dataset(dataset_name):
                raise DataLoadingError(
                    f"Invalid HuggingFace dataset pattern: {dataset_name}")

            dataset = load_dataset(dataset_name, split="train")
            return [dict(row) for row in dataset.select(range(min(len(dataset), max_samples)))]
        except Exception as e:
            raise DataLoadingError(
                f"Error loading HuggingFace dataset {dataset_name}: {e}")

    def _load_csv(self, file_path: str, max_samples: int) -> List[Dict]:
        try:
            if not os.path.exists(file_path):
                raise DataLoadingError(f"CSV file not found: {file_path}")
            df = pd.read_csv(file_path, nrows=max_samples)
            return df.to_dict('records')
        except Exception as e:
            raise DataLoadingError(f"Error loading CSV {file_path}: {e}")

    def _load_json(self, file_path: str, max_samples: int) -> List[Dict]:
        try:
            if not os.path.exists(file_path):
                raise DataLoadingError(f"JSON file not found: {file_path}")
            with open(file_path, 'r') as f:
                data = json.load(f)
            return data[:max_samples] if isinstance(data, list) else [data]
        except Exception as e:
            raise DataLoadingError(f"Error loading JSON {file_path}: {e}")

    def _detect_source_type(self, source: str) -> str:
        if source.endswith('.csv'):
            return DataSourceType.CSV.value
        elif source.endswith('.json'):
            return DataSourceType.JSON.value
        elif self._is_valid_hf_dataset(source):
            return DataSourceType.HUGGINGFACE.value
        else:
            return DataSourceType.UNKNOWN.value

    def _is_valid_hf_dataset(self, source: str) -> bool:
        """Check if source matches HuggingFace dataset pattern"""
        # Pattern: org_name/dataset_name (no spaces, valid characters)
        pattern = r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$'
        return bool(re.match(pattern, source)) and not source.startswith('http')

    def _create_cache_key(self, sample_data: Dict) -> str:
        field_names = sorted(sample_data.keys())
        return "|".join(field_names)

    def _normalize_row(self, row: Dict, field_mapping: FieldMapping) -> Optional[Dict[str, Any]]:
        try:
            transcript = str(
                row.get(field_mapping.transcript_field, "")).strip()
            reference_notes = str(
                row.get(field_mapping.reference_notes_field, "")).strip()

            if not transcript:
                return None

            patient_metadata = {}
            for field in field_mapping.patient_metadata_fields:
                if field in row and row[field] is not None:
                    patient_metadata[field] = row[field]

            normalized_row = {
                'transcript': transcript,
                'reference_notes': reference_notes,
                'patient_metadata': patient_metadata,
                'source': "auto_detected_mapping",
                'field_mapping_confidence': field_mapping.confidence_score
            }

            return normalized_row
        except Exception as e:
            print(f"Error normalizing row: {e}")
            return None
