"""
Flexible Storage System for SOAP Notes and Evaluation Metrics
==============================================================

Auto-detecting storage that saves SOAP generation results and evaluation metrics
with configurable modes and duplicate detection.
"""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from enum import Enum


class StorageMode(Enum):
    """
    Storage modes defining what data to save.

    SOAP_ONLY: Save only SOAP generation results
    EVALUATION_ONLY: Save only evaluation metrics
    BOTH: Save both SOAP results and evaluation metrics
    """
    SOAP_ONLY = "soap_only"
    EVALUATION_ONLY = "evaluation_only"
    BOTH = "both"


class FlexibleSOAPStorage:
    """
    Auto-detecting storage system for SOAP notes and evaluations.

    Handles duplicate detection, configurable storage modes, and automatic
    saving to JSON files. Filters stored data based on the selected mode.
    """

    def __init__(self, storage_file: str = "soap_results.json", mode: str = "both"):
        """
        Initialize storage with file path and mode.

        Args:
            storage_file: Path to JSON file for storing results
            mode: Storage mode ("soap_only", "evaluation_only", or "both")
        """
        self.storage_file = storage_file
        self.mode = StorageMode(mode)
        self.processed_hashes = set()
        self.results_data = []

        self._ensure_directory_exists()
        self._load_existing_data()

    def _ensure_directory_exists(self) -> None:
        """
        Create directory for storage file if it doesn't exist.

        Raises:
            Exception: If directory creation fails
        """
        directory = os.path.dirname(self.storage_file)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise Exception(f"Could not create directory {directory}: {e}")

    def _load_existing_data(self) -> None:
        """
        Load existing results from JSON file into memory.

        Populates processed_hashes set for duplicate detection and
        results_data list with existing records.
        """
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.results_data = json.load(f)

                # Extract hashes from existing data for duplicate detection
                for result in self.results_data:
                    if 'input_hash' in result:
                        self.processed_hashes.add(result['input_hash'])

            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
                self.results_data = []
                self.processed_hashes = set()

    def _save_all_data(self) -> bool:
        """
        Save all results data to JSON file.

        Returns:
            True if save was successful, False otherwise
        """
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.results_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data to file: {e}")
            return False

    def _create_input_hash(self, transcript: str, metadata: str) -> str:
        """
        Create MD5 hash of input data for duplicate detection.

        Args:
            transcript: Patient conversation transcript
            metadata: Patient metadata (as string)

        Returns:
            MD5 hash string of combined input
        """
        combined = f"{transcript}|{metadata}"
        return hashlib.md5(combined.encode()).hexdigest()

    def is_duplicate(self, transcript: str, metadata: str) -> bool:
        """
        Check if this input has already been processed.

        Args:
            transcript: Patient conversation transcript
            metadata: Patient metadata (as string)

        Returns:
            True if this input is a duplicate, False otherwise
        """
        input_hash = self._create_input_hash(transcript, str(metadata))
        return input_hash in self.processed_hashes

    def _filter_by_mode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Filter result dictionary based on storage mode.

        Removes unnecessary fields depending on whether we're storing
        SOAP results, evaluation metrics, or both.

        Args:
            result: Complete result dictionary

        Returns:
            Filtered dictionary containing only relevant fields for current mode
        """
        filtered = {}

        # Include basic metadata based on mode
        if self.mode == StorageMode.EVALUATION_ONLY:
            # Minimal metadata for evaluation only
            basic_fields = {'timestamp', 'input_hash'}
        else:
            # Full metadata for other modes (patient_metadata removed per user request)
            basic_fields = {'timestamp', 'input_hash', 'source_name',
                            'original_transcript', 'conversation'}

        for field in basic_fields:
            if field in result:
                filtered[field] = result[field]

        if self.mode == StorageMode.SOAP_ONLY:
            # Only SOAP generation results
            soap_fields = {'generated_soap_note', 'subjective_components', 'objective_components',
                           'assessment_components', 'plan_components', 'engine_info', 'engine_type',
                           'pipeline_info'}
            for field in soap_fields:
                if field in result:
                    filtered[field] = result[field]

        elif self.mode == StorageMode.EVALUATION_ONLY:
            # Only essential evaluation data - no personal info
            essential_fields = [
                'original_transcript', 'conversation', 'reference_notes', 'evaluation_metrics', 'source_name']
            for field in essential_fields:
                if field in result:
                    filtered[field] = result[field]

            # Also capture individual evaluation fields at top level
            eval_indicators = ['score', 'accuracy', 'coverage', 'completeness', 'validity',
                               'missing', 'hallucination', 'clinical']
            for key, value in result.items():
                if any(indicator in key.lower() for indicator in eval_indicators):
                    filtered[key] = value

        else:  # BOTH mode
            # Save everything - let the generators decide what to include
            filtered = result.copy()

        # Always remove patient metadata from stored results per user request
        filtered.pop('patient_metadata', None)

        return filtered

    def save_result(self, result: Dict[str, Any]) -> bool:
        """
        Save a single result with duplicate detection and mode filtering.

        Args:
            result: Result dictionary containing 'original_transcript' or 'conversation' key

        Returns:
            True if saved successfully, False if duplicate or save failed
        """
        try:
            # Handle both 'original_transcript' and 'conversation' keys
            transcript = result.get(
                'original_transcript') or result.get('conversation', '')
            if not transcript:
                raise KeyError(
                    "Result must contain 'original_transcript' or 'conversation' key")

            input_hash = self._create_input_hash(
                transcript,
                # Use for hashing but don't store
                str(result.get('patient_metadata', ''))
            )

            if input_hash in self.processed_hashes:
                return False  # Duplicate detected, skip silently

            # Add metadata
            result['timestamp'] = datetime.now().isoformat()
            result['input_hash'] = input_hash

            # Filter based on mode
            filtered_result = self._filter_by_mode(result)

            # Add to results data
            self.results_data.append(filtered_result)
            self.processed_hashes.add(input_hash)

            # Save all data to file
            return self._save_all_data()

        except Exception as e:
            print(f"Error saving result: {e}")
            return False

    def switch_mode(self, new_mode: str) -> None:
        """
        Switch to a different storage mode.

        Args:
            new_mode: New mode ("soap_only", "evaluation_only", or "both")
        """
        self.mode = StorageMode(new_mode)

    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all stored results from memory.

        Returns:
            Copy of all stored result dictionaries
        """
        return self.results_data.copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about stored results.

        Returns:
            Dictionary containing:
            - total_results: Number of stored results
            - storage_mode: Current storage mode
            - unique_sources: Number of unique data sources
            - first_processed: Timestamp of first result
            - last_processed: Timestamp of last result
            - results_with_soap: Count of results with SOAP notes
            - results_with_evaluation: Count of results with evaluations
            - detected_evaluation_fields: List of evaluation metric fields found
        """
        results = self.results_data
        if not results:
            return {"total_results": 0}

        stats = {
            "total_results": len(results),
            "storage_mode": self.mode.value,
            "unique_sources": len(set(r.get('source_name', 'unknown') for r in results)),
            "first_processed": results[0].get('timestamp', 'unknown') if results else 'unknown',
            "last_processed": results[-1].get('timestamp', 'unknown') if results else 'unknown'
        }

        # Count what types of data we have
        soap_count = sum(1 for r in results if 'generated_soap_note' in r)
        eval_count = sum(1 for r in results if 'evaluation_metrics' in r)

        stats.update({
            "results_with_soap": soap_count,
            "results_with_evaluation": eval_count,
        })

        # Auto-detect evaluation metrics that were saved
        all_eval_fields = set()
        for result in results:
            if 'evaluation_metrics' in result:
                all_eval_fields.update(result['evaluation_metrics'].keys())

            # Also check top-level evaluation fields
            eval_indicators = ['score', 'accuracy',
                               'coverage', 'completeness', 'validity']
            for key in result.keys():
                if any(indicator in key.lower() for indicator in eval_indicators):
                    all_eval_fields.add(key)

        stats["detected_evaluation_fields"] = sorted(list(all_eval_fields))

        return stats


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_soap_storage(mode: str = "both") -> FlexibleSOAPStorage:
    """
    Create storage with specified mode.

    Args:
        mode: Storage mode ("soap_only", "evaluation_only", or "both")

    Returns:
        Configured FlexibleSOAPStorage instance
    """
    return FlexibleSOAPStorage(mode=mode)


def create_soap_only_storage() -> FlexibleSOAPStorage:
    """
    Create storage that only saves SOAP generation results.

    Returns:
        FlexibleSOAPStorage configured for SOAP-only mode
    """
    return FlexibleSOAPStorage(mode="soap_only")


def create_evaluation_only_storage() -> FlexibleSOAPStorage:
    """
    Create storage that only saves evaluation results.

    Returns:
        FlexibleSOAPStorage configured for evaluation-only mode
    """
    return FlexibleSOAPStorage(mode="evaluation_only")


def create_full_storage() -> FlexibleSOAPStorage:
    """
    Create storage that saves everything.

    Returns:
        FlexibleSOAPStorage configured to save both SOAP and evaluation data
    """
    return FlexibleSOAPStorage(mode="both")
