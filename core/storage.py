"""Flexible storage system that auto-detects and saves evaluation metrics."""

import json
import os
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Set, Optional
from enum import Enum


class StorageMode(Enum):
    SOAP_ONLY = "soap_only"
    EVALUATION_ONLY = "evaluation_only"
    BOTH = "both"


class FlexibleSOAPStorage:
    """Auto-detecting storage that saves what you generate"""

    def __init__(self, storage_file: str = "soap_results.json", mode: str = "both"):
        self.storage_file = storage_file
        self.mode = StorageMode(mode)
        self.processed_hashes = set()
        self.results_data = []

        print(f"Storage mode: {self.mode.value}")
        self._ensure_directory_exists()
        self._load_existing_data()

    def _ensure_directory_exists(self):
        """Ensure the directory for storage file exists"""
        directory = os.path.dirname(self.storage_file)
        if directory:
            try:
                os.makedirs(directory, exist_ok=True)
            except Exception as e:
                raise Exception(f"Could not create directory {directory}: {e}")

    def _load_existing_data(self):
        """Load existing data from JSON file"""
        if os.path.exists(self.storage_file):
            try:
                with open(self.storage_file, 'r') as f:
                    self.results_data = json.load(f)

                # Extract hashes from existing data
                for result in self.results_data:
                    if 'input_hash' in result:
                        self.processed_hashes.add(result['input_hash'])

                print(f"Loaded {len(self.results_data)} existing results")
            except Exception as e:
                print(f"Warning: Could not load existing data: {e}")
                self.results_data = []
                self.processed_hashes = set()

    def _save_all_data(self):
        """Save all results data to JSON file"""
        try:
            with open(self.storage_file, 'w') as f:
                json.dump(self.results_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving data to file: {e}")
            return False

    def _create_input_hash(self, transcript: str, metadata: str) -> str:
        """Create hash of input data to detect duplicates"""
        combined = f"{transcript}|{metadata}"
        return hashlib.md5(combined.encode()).hexdigest()

    def is_already_processed(self, transcript: str, metadata: str) -> bool:
        """Check if this input has already been processed"""
        input_hash = self._create_input_hash(transcript, metadata)
        return input_hash in self.processed_hashes

    def _filter_by_mode(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filter result based on storage mode"""
        filtered = {}

        # Include basic metadata based on mode
        if self.mode == StorageMode.EVALUATION_ONLY:
            # Minimal metadata for evaluation only
            basic_fields = {'timestamp', 'input_hash'}
        else:
            # Full metadata for other modes
            basic_fields = {'timestamp', 'input_hash', 'source_name',
                            'original_transcript', 'patient_metadata'}

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
                'original_transcript', 'reference_notes', 'evaluation_metrics', 'source_name']
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

        return filtered

    def save_result(self, result: Dict[str, Any]) -> bool:
        """Save result with auto-detection based on mode"""
        try:
            input_hash = self._create_input_hash(
                result['original_transcript'],
                str(result['patient_metadata'])
            )

            if input_hash in self.processed_hashes:
                print(
                    f"Warning: Duplicate detected, not saving result with hash {input_hash[:8]}...")
                return False

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

    def switch_mode(self, new_mode: str):
        """Switch storage mode"""
        self.mode = StorageMode(new_mode)
        print(f"Storage mode switched to: {self.mode.value}")

    def load_all_results(self) -> List[Dict[str, Any]]:
        """Load all stored results"""
        return self.results_data.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about stored results"""
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
# SIMPLE USAGE
# =============================================================================

def create_soap_storage(mode: str = "both") -> FlexibleSOAPStorage:
    """Create storage with specified mode"""
    return FlexibleSOAPStorage(mode=mode)


def create_soap_only_storage() -> FlexibleSOAPStorage:
    """Create storage that only saves SOAP generation results"""
    return FlexibleSOAPStorage(mode="soap_only")


def create_evaluation_only_storage() -> FlexibleSOAPStorage:
    """Create storage that only saves evaluation results"""
    return FlexibleSOAPStorage(mode="evaluation_only")


def create_full_storage() -> FlexibleSOAPStorage:
    """Create storage that saves everything"""
    return FlexibleSOAPStorage(mode="both")
