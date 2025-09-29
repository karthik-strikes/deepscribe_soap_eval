"""
Precision/Recall LLM Evaluators with True Async/Batch Support
==============================================================

Medical SOAP note evaluation system with deterministic and LLM-based evaluators.
Supports true batch processing and comprehensive quality metrics.
"""

import json
import dspy
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from enum import Enum
from tqdm.asyncio import tqdm as async_tqdm
from utils.json_parser import safe_json_parse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluatorType(Enum):
    """
    Types of evaluators available.

    DETERMINISTIC: Fast rule-based evaluators
    LLM_JUDGE: Slower but more nuanced LLM-based evaluators
    """
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"


# ==================== BASE INTERFACE ====================

class BaseEvaluatorProtocol:
    """
    Protocol defining evaluator interface.

    All evaluators should implement evaluate_async and evaluate_batch_async methods.
    This is a documentation-only class, not used for inheritance.
    """
    pass


@dataclass
class PrecisionRecallMetrics:
    """
    Container for precision/recall metrics with detailed counts and lists.

    Tracks content fidelity (how well the note captures transcript information)
    and medical correctness (accuracy of medical statements).
    """

    correctly_captured: int = 0
    missed_critical: int = 0
    unsupported_content: int = 0
    medically_sound: int = 0
    medically_incorrect: int = 0

    correctly_captured_list: List[str] = field(default_factory=list)
    missed_critical_list: List[str] = field(default_factory=list)
    unsupported_content_list: List[str] = field(default_factory=list)
    medically_sound_list: List[str] = field(default_factory=list)
    medically_incorrect_list: List[str] = field(default_factory=list)

    @property
    def content_fidelity_recall(self) -> float:
        """
        Calculate recall for content fidelity (0-1 scale).

        Recall = TP / (TP + FN) = correctly_captured / (correctly_captured + missed_critical)
        Measures: What proportion of critical findings were captured?

        Returns:
            Recall score between 0 and 1
        """
        total_should_capture = self.correctly_captured + self.missed_critical
        if total_should_capture == 0:
            return 1.0
        return self.correctly_captured / total_should_capture

    @property
    def content_fidelity_precision(self) -> float:
        """
        Calculate precision for content fidelity (0-1 scale).

        Precision = TP / (TP + FP) = correctly_captured / (correctly_captured + unsupported_content)
        Measures: What proportion of captured content is accurate?

        Returns:
            Precision score between 0 and 1
        """
        total_captured = self.correctly_captured + self.unsupported_content
        if total_captured == 0:
            return 1.0
        return self.correctly_captured / total_captured

    @property
    def content_fidelity_f1(self) -> float:
        """
        Calculate F1 score for content fidelity (0-1 scale).

        F1 = 2 * (precision * recall) / (precision + recall)
        Harmonic mean of precision and recall.

        Returns:
            F1 score between 0 and 1
        """
        if self.content_fidelity_precision + self.content_fidelity_recall == 0:
            return 0.0
        return 2 * (self.content_fidelity_precision * self.content_fidelity_recall) / \
            (self.content_fidelity_precision + self.content_fidelity_recall)

    @property
    def medical_correctness_accuracy(self) -> float:
        """
        Calculate medical correctness accuracy (0-1 scale).

        Accuracy = correct / (correct + incorrect)
        Measures: What proportion of medical statements are accurate?

        Returns:
            Accuracy score between 0 and 1
        """
        total_statements = self.medically_sound + self.medically_incorrect
        if total_statements == 0:
            return 1.0
        return self.medically_sound / total_statements

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metrics to dictionary format for serialization.

        Returns:
            Dictionary with all metrics, counts, and detailed lists
        """
        return {
            'content_fidelity': {
                'recall': self.content_fidelity_recall,
                'precision': self.content_fidelity_precision,
                'f1': self.content_fidelity_f1,
                'counts': {
                    'correctly_captured': self.correctly_captured,
                    'missed_critical': self.missed_critical,
                    'unsupported_content': self.unsupported_content
                },
                'details': {
                    'correctly_captured_list': self.correctly_captured_list,
                    'missed_critical_list': self.missed_critical_list,
                    'unsupported_content_list': self.unsupported_content_list
                }
            },
            'medical_correctness': {
                'accuracy': self.medical_correctness_accuracy,
                'counts': {
                    'medically_sound': self.medically_sound,
                    'medically_incorrect': self.medically_incorrect
                },
                'details': {
                    'medically_sound_list': self.medically_sound_list,
                    'medically_incorrect_list': self.medically_incorrect_list
                }
            }
        }


# ==================== DSPY SIGNATURES ====================

# class ExtractCriticalFindings(dspy.Signature):
#     """Extract critical medical findings that must be documented"""
#     transcript: str = dspy.InputField(desc="Patient conversation transcript")
#     patient_metadata: str = dspy.InputField(
#         desc="Patient demographics and background information")
#     critical_findings: str = dspy.OutputField(
#         desc="JSON list of critical medical facts that must be captured in any clinical note"
#     )


# class ValidateContentFidelity(dspy.Signature):
#     """Validate if note content is faithful to transcript"""
#     critical_findings: str = dspy.InputField(
#         desc="JSON list of critical findings from transcript")
#     generated_note: str = dspy.InputField(
#         desc="Generated medical note to validate")
#     patient_metadata: str = dspy.InputField(
#         desc="Patient demographics and background information")
#     correctly_captured: str = dspy.OutputField(
#         desc="JSON object with 'list' and 'count' of critical findings correctly captured in note"
#     )
#     missed_critical: str = dspy.OutputField(
#         desc="JSON object with 'list' and 'count' of critical findings missing from note"
#     )
#     unsupported_content: str = dspy.OutputField(
#         desc="JSON object with 'list' and 'count' of medical content in note not supported by transcript or patient context"
#     )


# class ExtractMedicalStatements(dspy.Signature):
#     """Extract all medical statements from generated note"""
#     generated_note: str = dspy.InputField(desc="Generated medical note")
#     medical_statements: str = dspy.OutputField(
#         desc="JSON list of all medical statements, claims, and conclusions in the note"
#     )


# class ValidateMedicalAccuracy(dspy.Signature):
#     """Validate medical accuracy of statements"""
#     medical_statements: str = dspy.InputField(
#         desc="JSON list of medical statements to validate")
#     transcript: str = dspy.InputField(desc="Original transcript for context")
#     patient_metadata: str = dspy.InputField(
#         desc="Patient demographics and background information")
#     medically_sound: str = dspy.OutputField(
#         desc="JSON object with 'list' and 'count' of medically accurate and appropriate statements"
#     )
#     medically_incorrect: str = dspy.OutputField(
#         desc="JSON object with 'list' and 'count' of medically incorrect, inappropriate or misleading statements"
#     )


class ExtractCriticalFindings(dspy.Signature):
    """Extract critical medical findings that must be documented"""
    transcript: str = dspy.InputField(
        desc="Patient conversation transcript or reference SOAP note")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")
    critical_findings: str = dspy.OutputField(
        desc='JSON list of critical medical facts that must be captured in any clinical note. Example: ["Patient reports chest pain for 2 hours", "Blood pressure 160/95", "Family history of heart disease"]'
    )


class ValidateContentFidelity(dspy.Signature):
    """Validate if note content is faithful to transcript"""
    critical_findings: str = dspy.InputField(
        desc="JSON list of critical findings from transcript")
    generated_note: str = dspy.InputField(
        desc="Generated or reference medical note to validate")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")
    correctly_captured: str = dspy.OutputField(
        desc='JSON object with list and count of critical findings correctly captured in note. Example: {"list": ["Chest pain documented", "BP recorded"], "count": 2}'
    )
    missed_critical: str = dspy.OutputField(
        desc='JSON object with list and count of critical findings missing from note. Example: {"list": ["Family history not mentioned"], "count": 1}'
    )
    unsupported_content: str = dspy.OutputField(
        desc='JSON object with list and count of medical content in note not supported by transcript or patient context. Example: {"list": ["Mentions diabetes without evidence"], "count": 1}'
    )


class ExtractMedicalStatements(dspy.Signature):
    """Extract all medical statements from generated note"""
    generated_note: str = dspy.InputField(
        desc="Generated or reference medical note")
    medical_statements: str = dspy.OutputField(
        desc='JSON list of all medical statements, claims, and conclusions in the note. Example: ["Diagnosis: Acute bronchitis", "Prescribed amoxicillin 500mg", "Patient advised to rest"]'
    )


class ValidateMedicalAccuracy(dspy.Signature):
    """Validate medical accuracy of statements"""
    medical_statements: str = dspy.InputField(
        desc="JSON list of medical statements to validate")
    transcript: str = dspy.InputField(
        desc="Original transcript or reference SOAP note for context")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")
    medically_sound: str = dspy.OutputField(
        desc='JSON object with list and count of medically accurate and appropriate statements. Example: {"list": ["Appropriate antibiotic choice", "Correct dosage"], "count": 2}'
    )
    medically_incorrect: str = dspy.OutputField(
        desc='JSON object with list and count of medically incorrect, inappropriate or misleading statements. Example: {"list": ["Contraindicated for patient age"], "count": 1}'
    )

# ==================== DSPY MODULE IMPLEMENTATIONS ====================


class ContentFidelityEvaluator(dspy.Module):
    """
    Evaluates how faithfully a SOAP note captures information from the transcript.

    Uses two-step process:
    1. Extract critical findings from transcript that must be documented
    2. Validate which findings are captured vs missed in the generated note

    Supports true batch processing via DSPy's native batch() method.
    """

    def __init__(self):
        """Initialize the content fidelity evaluator with DSPy modules."""
        super().__init__()
        self.extract_ground_truth = dspy.ChainOfThought(
            ExtractCriticalFindings)
        self.validate_content = dspy.ChainOfThought(ValidateContentFidelity)

    def get_type(self) -> EvaluatorType:
        """Return evaluator type."""
        return EvaluatorType.LLM_JUDGE

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Evaluate a single note asynchronously.

        Args:
            transcript: Original patient-provider conversation
            generated_note: Generated SOAP note to evaluate
            patient_metadata: Patient demographics and background

        Returns:
            Dictionary with fidelity metrics (recall, precision, F1, counts, details)
        """
        try:
            return await asyncio.to_thread(self.forward, transcript,
                                           generated_note, patient_metadata)
        except Exception as e:
            logger.error(f"Content fidelity evaluation failed: {e}")
            return self._error_result(str(e))

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate multiple notes using true batch processing.

        Uses DSPy's native batch() method for efficient parallel processing.

        Args:
            transcripts: List of patient-provider conversations
            generated_notes: List of generated SOAP notes
            patient_metadata_list: List of patient metadata

        Returns:
            List of evaluation dictionaries with metrics for each note
        """
        try:
            logger.info(
                f"Starting batch content fidelity evaluation for {len(transcripts)} notes")

            # Step 1: Extract critical findings from all transcripts in batch
            extraction_examples = [
                dspy.Example(transcript=t, patient_metadata=m).with_inputs(
                    "transcript", "patient_metadata")
                for t, m in zip(transcripts, patient_metadata_list)
            ]

            extraction_results = await asyncio.to_thread(
                self.extract_ground_truth.batch,
                examples=extraction_examples,
                num_threads=min(len(transcripts), 10),
                max_errors=None,
                return_failed_examples=False
            )

            # Step 2: Validate content for all notes in batch
            validation_examples = [
                dspy.Example(
                    critical_findings=ext_result.critical_findings,
                    generated_note=note,
                    patient_metadata=metadata
                ).with_inputs("critical_findings", "generated_note", "patient_metadata")
                for ext_result, note, metadata in zip(extraction_results, generated_notes, patient_metadata_list)
            ]

            validation_results = await asyncio.to_thread(
                self.validate_content.batch,
                examples=validation_examples,
                num_threads=min(len(validation_examples), 10),
                max_errors=None,
                return_failed_examples=False
            )

            # Process results
            final_results = []
            for i, validation_result in enumerate(validation_results):
                try:
                    correctly_captured_data = safe_json_parse(
                        validation_result.correctly_captured)
                    missed_critical_data = safe_json_parse(
                        validation_result.missed_critical)
                    unsupported_content_data = safe_json_parse(
                        validation_result.unsupported_content)

                    final_results.append({
                        'content_fidelity_recall': self._calculate_recall(
                            correctly_captured_data.get('count', 0),
                            missed_critical_data.get('count', 0)
                        ),
                        'content_fidelity_precision': self._calculate_precision(
                            correctly_captured_data.get('count', 0),
                            unsupported_content_data.get('count', 0)
                        ),
                        'content_fidelity_f1': self._calculate_f1(
                            correctly_captured_data.get('count', 0),
                            missed_critical_data.get('count', 0),
                            unsupported_content_data.get('count', 0)
                        ),
                        'content_fidelity_counts': {
                            'correctly_captured': correctly_captured_data.get('count', 0),
                            'missed_critical': missed_critical_data.get('count', 0),
                            'unsupported_content': unsupported_content_data.get('count', 0)
                        },
                        'content_fidelity_detail': {
                            'correctly_captured_list': correctly_captured_data.get('list', []),
                            'missed_critical_list': missed_critical_data.get('list', []),
                            'unsupported_content_list': unsupported_content_data.get('list', [])
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to process result {i}: {e}")
                    final_results.append(
                        self._error_result(f"Processing failed: {e}"))

            logger.info(
                f"Completed batch content fidelity evaluation: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Batch content fidelity evaluation failed: {e}")
            return [self._error_result(str(e))] * len(transcripts)

    def forward(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Synchronous forward pass for single evaluation.

        Args:
            transcript: Original conversation
            generated_note: Generated SOAP note
            patient_metadata: Patient information

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            extraction_result = self.extract_ground_truth(
                transcript=transcript,
                patient_metadata=patient_metadata
            )

            validation_result = self.validate_content(
                critical_findings=extraction_result.critical_findings,
                generated_note=generated_note,
                patient_metadata=patient_metadata
            )

            correctly_captured_data = safe_json_parse(
                validation_result.correctly_captured)
            missed_critical_data = safe_json_parse(
                validation_result.missed_critical)
            unsupported_content_data = safe_json_parse(
                validation_result.unsupported_content)

            return {
                'content_fidelity_recall': self._calculate_recall(
                    correctly_captured_data.get('count', 0),
                    missed_critical_data.get('count', 0)
                ),
                'content_fidelity_precision': self._calculate_precision(
                    correctly_captured_data.get('count', 0),
                    unsupported_content_data.get('count', 0)
                ),
                'content_fidelity_f1': self._calculate_f1(
                    correctly_captured_data.get('count', 0),
                    missed_critical_data.get('count', 0),
                    unsupported_content_data.get('count', 0)
                ),
                'content_fidelity_counts': {
                    'correctly_captured': correctly_captured_data.get('count', 0),
                    'missed_critical': missed_critical_data.get('count', 0),
                    'unsupported_content': unsupported_content_data.get('count', 0)
                },
                'content_fidelity_detail': {
                    'correctly_captured_list': correctly_captured_data.get('list', []),
                    'missed_critical_list': missed_critical_data.get('list', []),
                    'unsupported_content_list': unsupported_content_data.get('list', [])
                }
            }

        except Exception as e:
            logger.error(f"Content fidelity forward pass failed: {e}")
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'content_fidelity_recall': 0.0,
            'content_fidelity_precision': 0.0,
            'content_fidelity_f1': 0.0,
            'content_fidelity_counts': {'correctly_captured': 0, 'missed_critical': 0, 'unsupported_content': 0},
            'content_fidelity_detail': {'error': error_msg}
        }

    def _calculate_recall(self, correctly_captured: int, missed_critical: int) -> float:
        """Calculate recall metric."""
        total_should_capture = correctly_captured + missed_critical
        if total_should_capture == 0:
            return 1.0
        return correctly_captured / total_should_capture

    def _calculate_precision(self, correctly_captured: int, unsupported_content: int) -> float:
        """Calculate precision metric."""
        total_captured = correctly_captured + unsupported_content
        if total_captured == 0:
            return 1.0
        return correctly_captured / total_captured

    def _calculate_f1(self, correctly_captured: int, missed_critical: int, unsupported_content: int) -> float:
        """Calculate F1 score."""
        recall = self._calculate_recall(correctly_captured, missed_critical)
        precision = self._calculate_precision(
            correctly_captured, unsupported_content)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class MedicalCorrectnessEvaluator(dspy.Module):
    """
    Evaluates medical accuracy of statements in a SOAP note.

    Uses two-step process:
    1. Extract all medical statements from the generated note
    2. Validate which statements are medically sound vs incorrect

    Supports true batch processing via DSPy's native batch() method.
    """

    def __init__(self):
        """Initialize the medical correctness evaluator with DSPy modules."""
        super().__init__()
        self.extract_statements = dspy.ChainOfThought(ExtractMedicalStatements)
        self.validate_accuracy = dspy.ChainOfThought(ValidateMedicalAccuracy)

    def get_type(self) -> EvaluatorType:
        """Return evaluator type."""
        return EvaluatorType.LLM_JUDGE

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Evaluate medical correctness of a single note asynchronously.

        Args:
            transcript: Original patient-provider conversation
            generated_note: Generated SOAP note to evaluate
            patient_metadata: Patient demographics and background

        Returns:
            Dictionary with medical correctness metrics (accuracy, counts, details)
        """
        try:
            return await asyncio.to_thread(self.forward, transcript,
                                           generated_note, patient_metadata)
        except Exception as e:
            logger.error(f"Medical correctness evaluation failed: {e}")
            return self._error_result(str(e))

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Evaluate medical correctness of multiple notes using true batch processing.

        Args:
            transcripts: List of patient-provider conversations
            generated_notes: List of generated SOAP notes
            patient_metadata_list: List of patient metadata

        Returns:
            List of evaluation dictionaries with metrics for each note
        """
        try:
            logger.info(
                f"Starting batch medical correctness evaluation for {len(generated_notes)} notes")

            # Step 1: Extract medical statements from all notes in batch
            extraction_examples = [
                dspy.Example(generated_note=note).with_inputs("generated_note")
                for note in generated_notes
            ]

            extraction_results = await asyncio.to_thread(
                self.extract_statements.batch,
                examples=extraction_examples,
                num_threads=min(len(generated_notes), 10),
                max_errors=None,
                return_failed_examples=False
            )

            # Step 2: Validate accuracy for all statements in batch
            validation_examples = [
                dspy.Example(
                    medical_statements=ext_result.medical_statements,
                    transcript=transcript,
                    patient_metadata=metadata
                ).with_inputs("medical_statements", "transcript", "patient_metadata")
                for ext_result, transcript, metadata in zip(extraction_results, transcripts, patient_metadata_list)
            ]

            validation_results = await asyncio.to_thread(
                self.validate_accuracy.batch,
                examples=validation_examples,
                num_threads=min(len(validation_examples), 10),
                max_errors=None,
                return_failed_examples=False
            )

            # Process results
            final_results = []
            for i, validation_result in enumerate(validation_results):
                try:
                    medically_sound_data = safe_json_parse(
                        validation_result.medically_sound)
                    medically_incorrect_data = safe_json_parse(
                        validation_result.medically_incorrect)

                    final_results.append({
                        'medical_correctness_accuracy': self._calculate_accuracy(
                            medically_sound_data.get('count', 0),
                            medically_incorrect_data.get('count', 0)
                        ),
                        'medical_correctness_counts': {
                            'medically_sound': medically_sound_data.get('count', 0),
                            'medically_incorrect': medically_incorrect_data.get('count', 0)
                        },
                        'medical_correctness_detail': {
                            'medically_sound_list': medically_sound_data.get('list', []),
                            'medically_incorrect_list': medically_incorrect_data.get('list', [])
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to process result {i}: {e}")
                    final_results.append(
                        self._error_result(f"Processing failed: {e}"))

            logger.info(
                f"Completed batch medical correctness evaluation: {len(final_results)} results")
            return final_results

        except Exception as e:
            logger.error(f"Batch medical correctness evaluation failed: {e}")
            return [self._error_result(str(e))] * len(transcripts)

    def forward(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Synchronous forward pass for single evaluation.

        Args:
            transcript: Original conversation
            generated_note: Generated SOAP note
            patient_metadata: Patient information

        Returns:
            Dictionary with evaluation metrics
        """
        try:
            extraction_result = self.extract_statements(
                generated_note=generated_note)
            validation_result = self.validate_accuracy(
                medical_statements=extraction_result.medical_statements,
                transcript=transcript,
                patient_metadata=patient_metadata
            )

            medically_sound_data = safe_json_parse(
                validation_result.medically_sound)
            medically_incorrect_data = safe_json_parse(
                validation_result.medically_incorrect)

            return {
                'medical_correctness_accuracy': self._calculate_accuracy(
                    medically_sound_data.get('count', 0),
                    medically_incorrect_data.get('count', 0)
                ),
                'medical_correctness_counts': {
                    'medically_sound': medically_sound_data.get('count', 0),
                    'medically_incorrect': medically_incorrect_data.get('count', 0)
                },
                'medical_correctness_detail': {
                    'medically_sound_list': medically_sound_data.get('list', []),
                    'medically_incorrect_list': medically_incorrect_data.get('list', [])
                }
            }

        except Exception as e:
            logger.error(f"Medical correctness forward pass failed: {e}")
            return self._error_result(str(e))

    def _error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'medical_correctness_accuracy': 1.0,
            'medical_correctness_counts': {'medically_sound': 0, 'medically_incorrect': 0},
            'medical_correctness_detail': {'error': error_msg}
        }

    def _calculate_accuracy(self, medically_sound: int, medically_incorrect: int) -> float:
        """Calculate accuracy metric."""
        total_statements = medically_sound + medically_incorrect
        if total_statements == 0:
            return 1.0
        return medically_sound / total_statements


# ==================== DETERMINISTIC EVALUATORS ====================

class EntityCoverageEvaluator:
    """
    Deterministic evaluator checking if key medical entities appear in the note.

    Uses regex patterns to detect:
    - Medications (drug names, dosages)
    - Symptoms (pain, fever, etc.)
    - Vital signs (BP, heart rate, etc.)
    - Procedures (x-ray, CT, etc.)

    NOTE: This is a basic implementation with limited patterns. For production
    use, consider integrating a medical NER library or expanding patterns.
    """

    def __init__(self):
        """Initialize with basic medical entity regex patterns."""
        self.medical_patterns = {
            'medications': r'\b(?:\w+(?:cillin|mycin|pril|statin|olol|pine|zole)|mg|tablet|capsule|pill)\b',
            'symptoms': r'\b(?:pain|ache|fever|nausea|vomiting|headache|dizzy|shortness of breath|chest pain|fatigue|weakness)\b',
            'vital_signs': r'\b(?:\d{2,3}/\d{2,3}|\d{2,3}\s*bpm|\d{2,3}°?[FC]|O2\s*sat|\d{1,3}%)\b',
            'procedures': r'\b(?:x-ray|ct scan|mri|ekg|ecg|blood test|biopsy|surgery)\b',
        }

    def get_type(self) -> EvaluatorType:
        """Return evaluator type."""
        return EvaluatorType.DETERMINISTIC

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Async wrapper for deterministic evaluation.

        Args:
            transcript: Original conversation
            generated_note: Generated SOAP note
            patient_metadata: Patient information (unused)

        Returns:
            Dictionary with entity_coverage percentage and missing_entities list
        """
        return self._evaluate_sync(transcript, generated_note, patient_metadata)

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Batch evaluation (runs synchronously since deterministic evaluation is fast).

        Args:
            transcripts: List of conversations
            generated_notes: List of SOAP notes
            patient_metadata_list: List of patient information

        Returns:
            List of evaluation dictionaries
        """
        return [
            self._evaluate_sync(t, n, m)
            for t, n, m in zip(transcripts, generated_notes, patient_metadata_list)
        ]

    def _evaluate_sync(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Internal synchronous evaluation logic.

        Extracts entities from transcript and checks if they appear in the note.
        """
        transcript_entities = self._extract_entities(transcript)
        note_entities = self._extract_entities(generated_note)

        total_entities = sum(len(entities)
                             for entities in transcript_entities.values())
        if total_entities == 0:
            return {'entity_coverage': 100.0, 'missing_entities': []}

        covered_entities = 0
        missing_entities = []

        for entity_type, transcript_set in transcript_entities.items():
            note_set = note_entities.get(entity_type, set())
            covered = len(transcript_set.intersection(note_set))
            covered_entities += covered

            missing = transcript_set - note_set
            missing_entities.extend(
                [f"{entity_type}: {entity}" for entity in missing])

        coverage = (covered_entities / total_entities) * 100

        return {
            'entity_coverage': coverage,
            'missing_entities': missing_entities
        }

    def _extract_entities(self, text: str) -> Dict[str, Set[str]]:
        """Extract medical entities from text using regex patterns."""
        import re
        entities = {}
        text_lower = text.lower()

        for entity_type, pattern in self.medical_patterns.items():
            matches = set(re.findall(pattern, text_lower, re.IGNORECASE))
            entities[entity_type] = matches

        return entities


class SOAPCompletenessEvaluator:
    """
    Deterministic evaluator checking if note has proper SOAP structure.

    Verifies presence of four required sections:
    - Subjective (chief complaint, HPI)
    - Objective (physical exam, vitals)
    - Assessment (diagnoses, clinical reasoning)
    - Plan (treatment, follow-up)
    """

    def __init__(self):
        """Initialize with required SOAP section patterns."""
        self.required_sections = {
            'subjective': r'(?:subjective|chief complaint|cc:|history of present illness|hpi)',
            'objective': r'(?:objective|physical exam|pe:|vital signs|vs:)',
            'assessment': r'(?:assessment|diagnosis|impression|dx:)',
            'plan': r'(?:plan|treatment|recommendations|follow.?up)'
        }

    def get_type(self) -> EvaluatorType:
        """Return evaluator type."""
        return EvaluatorType.DETERMINISTIC

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Async wrapper for SOAP completeness check.

        Args:
            transcript: Original conversation (unused)
            generated_note: Generated SOAP note to check
            patient_metadata: Patient information (unused)

        Returns:
            Dictionary with section_completeness percentage and missing_sections list
        """
        return self._evaluate_sync(transcript, generated_note, patient_metadata)

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Batch evaluation of SOAP completeness.

        Args:
            transcripts: List of conversations (unused)
            generated_notes: List of SOAP notes to check
            patient_metadata_list: List of patient information (unused)

        Returns:
            List of evaluation dictionaries
        """
        return [
            self._evaluate_sync(t, n, m)
            for t, n, m in zip(transcripts, generated_notes, patient_metadata_list)
        ]

    def _evaluate_sync(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Internal synchronous evaluation checking for required SOAP sections.

        Uses regex to detect section headers in the note.
        """
        import re
        note_lower = generated_note.lower()
        missing_sections = []
        present_sections = 0

        for section_name, pattern in self.required_sections.items():
            if re.search(pattern, note_lower, re.IGNORECASE):
                present_sections += 1
            else:
                missing_sections.append(section_name)

        score = (present_sections / len(self.required_sections)) * 100

        return {
            'section_completeness': score,
            'missing_sections': missing_sections
        }


class FormatValidityEvaluator:
    """
    Deterministic evaluator checking basic format requirements.

    Validates:
    - Note length (not too short or too long)
    - Proper sentence structure
    - Patient references present
    - No placeholder text

    Thresholds are configurable for different use cases.
    """

    def __init__(self, min_length: int = 50, max_length: int = 3000):
        """
        Initialize with configurable length thresholds.

        Args:
            min_length: Minimum acceptable note length in characters
            max_length: Maximum acceptable note length in characters
        """
        self.min_length = min_length
        self.max_length = max_length

    def get_type(self) -> EvaluatorType:
        """Return evaluator type."""
        return EvaluatorType.DETERMINISTIC

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Async wrapper for format validation.

        Args:
            transcript: Original conversation (unused)
            generated_note: Generated SOAP note to validate
            patient_metadata: Patient information (unused)

        Returns:
            Dictionary with format_validity percentage and format_issues list
        """
        return self._evaluate_sync(transcript, generated_note, patient_metadata)

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Batch evaluation of format validity.

        Args:
            transcripts: List of conversations (unused)
            generated_notes: List of SOAP notes to validate
            patient_metadata_list: List of patient information (unused)

        Returns:
            List of evaluation dictionaries
        """
        return [
            self._evaluate_sync(t, n, m)
            for t, n, m in zip(transcripts, generated_notes, patient_metadata_list)
        ]

    def _evaluate_sync(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """
        Internal synchronous validation checking basic format requirements.
        """
        import re
        issues = []
        score = 100.0

        # Check length
        note_length = len(generated_note.strip())
        if note_length < self.min_length:
            issues.append(
                f"Note too short ({note_length} chars, minimum {self.min_length})")
            score -= 30
        elif note_length > self.max_length:
            issues.append(
                f"Note too long ({note_length} chars, maximum {self.max_length})")
            score -= 10

        # Check sentence structure
        if not re.search(r'[.!?]', generated_note):
            issues.append("No proper sentence structure")
            score -= 20

        # Check patient references
        if not re.search(r'(?:patient|pt)', generated_note, re.IGNORECASE):
            issues.append("Missing patient references")
            score -= 15

        # Check for placeholders
        placeholders = ['[PLACEHOLDER]', 'TODO', 'FIXME', 'XXX']
        for placeholder in placeholders:
            if placeholder.lower() in generated_note.lower():
                issues.append(f"Contains placeholder: {placeholder}")
                score -= 25

        return {
            'format_validity': max(0, score),
            'format_issues': issues
        }


# ==================== METRICS DATACLASS ====================

@dataclass
class EnhancedEvaluationMetrics:
    """
    Container for all evaluation metrics.

    Combines deterministic and LLM-based metrics into a single dataclass
    for easy serialization and analysis.
    """

    # Deterministic metrics (0-100 scale)
    entity_coverage: float = 0.0
    section_completeness: float = 0.0
    format_validity: float = 0.0

    # LLM Judge metrics (0-1 scale)
    content_fidelity_recall: float = 0.0
    content_fidelity_precision: float = 0.0
    content_fidelity_f1: float = 0.0
    medical_correctness_accuracy: float = 0.0

    # Counts
    content_fidelity_counts: Dict[str, int] = field(default_factory=dict)
    medical_correctness_counts: Dict[str, int] = field(default_factory=dict)

    # Details
    missing_entities: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    llm_feedback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, mode: str = "comprehensive") -> Dict[str, Any]:
        """
        Convert metrics to dictionary format.

        Args:
            mode: Evaluation mode ("deterministic", "llm_only", "comprehensive")

        Returns:
            Dictionary organized by metric category (filtered by mode)
        """
        result = {}

        # Include deterministic metrics for deterministic and comprehensive modes
        if mode in ["deterministic", "comprehensive"]:
            result['deterministic_metrics'] = {
                'entity_coverage': self.entity_coverage,
                'section_completeness': self.section_completeness,
                'format_validity': self.format_validity,
            }

        # Include LLM metrics for llm_only and comprehensive modes
        if mode in ["llm_only", "comprehensive"]:
            result['llm_metrics'] = {
                'content_fidelity': {
                    'recall': self.content_fidelity_recall,
                    'precision': self.content_fidelity_precision,
                    'f1': self.content_fidelity_f1,
                    'counts': self.content_fidelity_counts
                },
                'medical_correctness': {
                    'accuracy': self.medical_correctness_accuracy,
                    'counts': self.medical_correctness_counts
                }
            }

        # Always include details (they're filtered based on what evaluators ran)
        details = {}
        if mode in ["deterministic", "comprehensive"]:
            details.update({
                'missing_entities': self.missing_entities,
                'missing_sections': self.missing_sections,
                'format_issues': getattr(self, 'format_issues', []),
            })
        if mode in ["llm_only", "comprehensive"]:
            details.update({
                'llm_feedback': self.llm_feedback,
            })

        if details:
            result['details'] = details

        return result

    def overall_quality_score(self) -> float:
        """
        Calculate weighted overall quality score (0-1 scale).

        Weights:
        - Content Fidelity F1: 40%
        - Medical Correctness: 30%
        - Entity Coverage: 10%
        - Section Completeness: 10%
        - Format Validity: 10%

        Returns:
            Overall quality score between 0 and 1
        """
        metrics = [
            self.content_fidelity_f1 * 0.4,
            self.medical_correctness_accuracy * 0.3,
            (self.entity_coverage / 100) * 0.1,
            (self.section_completeness / 100) * 0.1,
            (self.format_validity / 100) * 0.1
        ]
        return sum(metrics)


# ==================== REGISTRY ====================

class EvaluatorRegistry:
    """
    Registry for managing and creating evaluator instances.

    Provides centralized registration and factory methods for both
    deterministic and LLM-based evaluators.
    """

    _deterministic_evaluators: Dict[str, type] = {}
    _llm_evaluators: Dict[str, type] = {}

    @classmethod
    def register_deterministic(cls, name: str, evaluator_class: type) -> None:
        """Register a deterministic evaluator class."""
        cls._deterministic_evaluators[name] = evaluator_class

    @classmethod
    def register_llm(cls, name: str, evaluator_class: type) -> None:
        """Register an LLM evaluator class."""
        cls._llm_evaluators[name] = evaluator_class

    @classmethod
    def create_deterministic_evaluator(cls, name: str, **kwargs):
        """
        Create instance of a deterministic evaluator.

        Args:
            name: Registered evaluator name
            **kwargs: Arguments to pass to evaluator constructor

        Returns:
            Evaluator instance

        Raises:
            ValueError: If evaluator name not found
        """
        if name not in cls._deterministic_evaluators:
            raise ValueError(f"Unknown deterministic evaluator: {name}")
        return cls._deterministic_evaluators[name](**kwargs)

    @classmethod
    def create_llm_evaluator(cls, name: str, **kwargs):
        """
        Create instance of an LLM evaluator.

        Args:
            name: Registered evaluator name
            **kwargs: Arguments to pass to evaluator constructor

        Returns:
            Evaluator instance

        Raises:
            ValueError: If evaluator name not found
        """
        if name not in cls._llm_evaluators:
            raise ValueError(f"Unknown LLM evaluator: {name}")
        return cls._llm_evaluators[name](**kwargs)

    @classmethod
    def get_available_deterministic(cls) -> List[str]:
        """Get list of registered deterministic evaluator names."""
        return list(cls._deterministic_evaluators.keys())

    @classmethod
    def get_available_llm(cls) -> List[str]:
        """Get list of registered LLM evaluator names."""
        return list(cls._llm_evaluators.keys())


# Register all available evaluators
EvaluatorRegistry.register_deterministic(
    "entity_coverage", EntityCoverageEvaluator)
EvaluatorRegistry.register_deterministic(
    "soap_completeness", SOAPCompletenessEvaluator)
EvaluatorRegistry.register_deterministic(
    "format_validity", FormatValidityEvaluator)
EvaluatorRegistry.register_llm("content_fidelity", ContentFidelityEvaluator)
EvaluatorRegistry.register_llm(
    "medical_correctness", MedicalCorrectnessEvaluator)


# ==================== EVALUATION PIPELINE ====================

class EvaluationPipeline:
    """
    Main evaluation pipeline coordinating multiple evaluators.

    Supports three evaluation modes:
    - deterministic: Fast rule-based evaluation only
    - llm_only: Deep LLM-based analysis only  
    - comprehensive: Both deterministic and LLM evaluations

    Uses true batch processing for efficiency.
    """

    def __init__(self, deterministic_evaluators: Optional[List[str]] = None,
                 llm_evaluators: Optional[List[str]] = None):
        """
        Initialize pipeline with specific evaluators.

        Args:
            deterministic_evaluators: List of deterministic evaluator names to use
                                     (defaults to all available)
            llm_evaluators: List of LLM evaluator names to use
                           (defaults to all available)
        """
        det_names = deterministic_evaluators or EvaluatorRegistry.get_available_deterministic()
        llm_names = llm_evaluators or EvaluatorRegistry.get_available_llm()

        self.deterministic_evaluators = [
            EvaluatorRegistry.create_deterministic_evaluator(name) for name in det_names
        ]
        self.llm_evaluators = [
            EvaluatorRegistry.create_llm_evaluator(name) for name in llm_names
        ]

        logger.info(f"Initialized evaluation pipeline with {len(self.deterministic_evaluators)} deterministic "
                    f"and {len(self.llm_evaluators)} LLM evaluators")

    async def evaluate_async(self, transcript: str, generated_note: str,
                             patient_metadata: str, mode: str) -> Dict[str, Any]:
        """
        Evaluate a single SOAP note.

        Args:
            transcript: Original patient-provider conversation
            generated_note: Generated SOAP note to evaluate
            patient_metadata: Patient demographics and background
            mode: Evaluation mode ("deterministic", "llm_only", or "comprehensive")

        Returns:
            Dictionary with evaluation metrics
        """
        if mode == "deterministic":
            return await self._evaluate_deterministic_async(transcript, generated_note, patient_metadata)
        elif mode == "llm_only":
            return await self._evaluate_llm_only_async(transcript, generated_note, patient_metadata)
        else:  # comprehensive
            return await self._evaluate_comprehensive_async(transcript, generated_note, patient_metadata)

    async def evaluate_batch_async(self, transcripts: List[str], generated_notes: List[str],
                                   patient_metadata_list: List[str], mode: str) -> List[Dict[str, Any]]:
        """
        Evaluate multiple SOAP notes using true batch processing.

        Args:
            transcripts: List of patient-provider conversations
            generated_notes: List of generated SOAP notes
            patient_metadata_list: List of patient metadata
            mode: Evaluation mode ("deterministic", "llm_only", or "comprehensive")

        Returns:
            List of evaluation metric dictionaries
        """
        if mode == "deterministic":
            return await self._evaluate_deterministic_batch_async(transcripts, generated_notes, patient_metadata_list)
        elif mode == "llm_only":
            return await self._evaluate_llm_only_batch_async(transcripts, generated_notes, patient_metadata_list)
        else:  # comprehensive
            return await self._evaluate_comprehensive_batch_async(transcripts, generated_notes, patient_metadata_list)

    async def _evaluate_deterministic_async(self, transcript: str, generated_note: str,
                                            patient_metadata: str) -> Dict[str, Any]:
        """Single deterministic evaluation - fast rule-based checks."""
        results = {}
        for evaluator in self.deterministic_evaluators:
            eval_result = await evaluator.evaluate_async(transcript, generated_note, patient_metadata)
            results.update(eval_result)

        metrics = EnhancedEvaluationMetrics(
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
        )
        return metrics.to_dict("deterministic")

    async def _evaluate_deterministic_batch_async(self, transcripts: List[str],
                                                  generated_notes: List[str],
                                                  patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """Batch deterministic evaluation."""
        all_results = [{} for _ in range(len(transcripts))]

        for evaluator in self.deterministic_evaluators:
            batch_results = await evaluator.evaluate_batch_async(transcripts, generated_notes, patient_metadata_list)
            for i, result in enumerate(batch_results):
                all_results[i].update(result)

        final_metrics = []
        for results in all_results:
            metrics = EnhancedEvaluationMetrics(
                entity_coverage=results.get('entity_coverage', 0.0),
                section_completeness=results.get('section_completeness', 0.0),
                format_validity=results.get('format_validity', 0.0),
                missing_entities=results.get('missing_entities', []),
                missing_sections=results.get('missing_sections', []),
                format_issues=results.get('format_issues', []),
            )
            final_metrics.append(metrics.to_dict("deterministic"))

        return final_metrics

    async def _evaluate_llm_only_async(self, transcript: str, generated_note: str,
                                       patient_metadata: str) -> Dict[str, Any]:
        """Single LLM-only evaluation - deep analysis using language models."""
        results = {}
        llm_feedback = {}

        # Run all LLM evaluators in parallel
        tasks = [
            evaluator.evaluate_async(
                transcript, generated_note, patient_metadata)
            for evaluator in self.llm_evaluators
        ]

        eval_results = await asyncio.gather(*tasks)

        for eval_result in eval_results:
            results.update(eval_result)
            for key, value in eval_result.items():
                if key.endswith('_detail'):
                    llm_feedback[key] = value

        metrics = EnhancedEvaluationMetrics(
            content_fidelity_recall=results.get(
                'content_fidelity_recall', 0.0),
            content_fidelity_precision=results.get(
                'content_fidelity_precision', 0.0),
            content_fidelity_f1=results.get('content_fidelity_f1', 0.0),
            medical_correctness_accuracy=results.get(
                'medical_correctness_accuracy', 0.0),
            content_fidelity_counts=results.get('content_fidelity_counts', {}),
            medical_correctness_counts=results.get(
                'medical_correctness_counts', {}),
            llm_feedback=llm_feedback,
        )
        return metrics.to_dict("llm_only")

    async def _evaluate_llm_only_batch_async(self, transcripts: List[str],
                                             generated_notes: List[str],
                                             patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """TRUE BATCH: LLM-only evaluation using native batching."""
        # Run all LLM evaluators in parallel, each processing entire batch
        tasks = [
            evaluator.evaluate_batch_async(
                transcripts, generated_notes, patient_metadata_list)
            for evaluator in self.llm_evaluators
        ]

        all_eval_results = await asyncio.gather(*tasks)

        # Merge results for each item
        num_items = len(transcripts)
        final_metrics = []

        for i in range(num_items):
            results = {}
            llm_feedback = {}

            for eval_results_list in all_eval_results:
                item_result = eval_results_list[i]
                results.update(item_result)
                for key, value in item_result.items():
                    if key.endswith('_detail'):
                        llm_feedback[key] = value

            metrics = EnhancedEvaluationMetrics(
                content_fidelity_recall=results.get(
                    'content_fidelity_recall', 0.0),
                content_fidelity_precision=results.get(
                    'content_fidelity_precision', 0.0),
                content_fidelity_f1=results.get('content_fidelity_f1', 0.0),
                medical_correctness_accuracy=results.get(
                    'medical_correctness_accuracy', 0.0),
                content_fidelity_counts=results.get(
                    'content_fidelity_counts', {}),
                medical_correctness_counts=results.get(
                    'medical_correctness_counts', {}),
                llm_feedback=llm_feedback,
            )
            final_metrics.append(metrics.to_dict("llm_only"))

        return final_metrics

    async def _evaluate_comprehensive_async(self, transcript: str, generated_note: str,
                                            patient_metadata: str) -> Dict[str, Any]:
        """Single comprehensive evaluation - both deterministic and LLM."""
        results = {}
        llm_feedback = {}

        # Run deterministic evaluators (fast, sequential is fine)
        for evaluator in self.deterministic_evaluators:
            eval_result = await evaluator.evaluate_async(transcript, generated_note, patient_metadata)
            results.update(eval_result)

        # Run LLM evaluators in parallel
        if self.llm_evaluators:
            tasks = [
                evaluator.evaluate_async(
                    transcript, generated_note, patient_metadata)
                for evaluator in self.llm_evaluators
            ]
            eval_results = await asyncio.gather(*tasks)

            for eval_result in eval_results:
                results.update(eval_result)
                for key, value in eval_result.items():
                    if key.endswith('_detail'):
                        llm_feedback[key] = value

        metrics = EnhancedEvaluationMetrics(
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            content_fidelity_recall=results.get(
                'content_fidelity_recall', 0.0),
            content_fidelity_precision=results.get(
                'content_fidelity_precision', 0.0),
            content_fidelity_f1=results.get('content_fidelity_f1', 0.0),
            medical_correctness_accuracy=results.get(
                'medical_correctness_accuracy', 0.0),
            content_fidelity_counts=results.get('content_fidelity_counts', {}),
            medical_correctness_counts=results.get(
                'medical_correctness_counts', {}),
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
            llm_feedback=llm_feedback,
        )
        return metrics.to_dict("comprehensive")

    async def _evaluate_comprehensive_batch_async(self, transcripts: List[str],
                                                  generated_notes: List[str],
                                                  patient_metadata_list: List[str]) -> List[Dict[str, Any]]:
        """TRUE BATCH: Comprehensive evaluation with both evaluator types."""
        num_items = len(transcripts)
        all_results = [{} for _ in range(num_items)]
        all_llm_feedback = [{} for _ in range(num_items)]

        # Deterministic evaluators (batch)
        for evaluator in self.deterministic_evaluators:
            batch_results = await evaluator.evaluate_batch_async(transcripts, generated_notes, patient_metadata_list)
            for i, result in enumerate(batch_results):
                all_results[i].update(result)

        # LLM evaluators (parallel batch)
        if self.llm_evaluators:
            tasks = [
                evaluator.evaluate_batch_async(
                    transcripts, generated_notes, patient_metadata_list)
                for evaluator in self.llm_evaluators
            ]
            all_eval_results = await asyncio.gather(*tasks)

            for eval_results_list in all_eval_results:
                for i, item_result in enumerate(eval_results_list):
                    all_results[i].update(item_result)
                    for key, value in item_result.items():
                        if key.endswith('_detail'):
                            all_llm_feedback[i][key] = value

        # Build final metrics
        final_metrics = []
        for i in range(num_items):
            results = all_results[i]
            llm_feedback = all_llm_feedback[i]

            metrics = EnhancedEvaluationMetrics(
                entity_coverage=results.get('entity_coverage', 0.0),
                section_completeness=results.get('section_completeness', 0.0),
                format_validity=results.get('format_validity', 0.0),
                content_fidelity_recall=results.get(
                    'content_fidelity_recall', 0.0),
                content_fidelity_precision=results.get(
                    'content_fidelity_precision', 0.0),
                content_fidelity_f1=results.get('content_fidelity_f1', 0.0),
                medical_correctness_accuracy=results.get(
                    'medical_correctness_accuracy', 0.0),
                content_fidelity_counts=results.get(
                    'content_fidelity_counts', {}),
                medical_correctness_counts=results.get(
                    'medical_correctness_counts', {}),
                missing_entities=results.get('missing_entities', []),
                missing_sections=results.get('missing_sections', []),
                format_issues=results.get('format_issues', []),
                llm_feedback=llm_feedback,
            )
            final_metrics.append(metrics.to_dict("comprehensive"))

        return final_metrics


# ==================== FACTORY FUNCTIONS ====================

def create_evaluator(deterministic_evaluators: Optional[List[str]] = None,
                     llm_evaluators: Optional[List[str]] = None) -> EvaluationPipeline:
    """
    Create evaluation pipeline with specific evaluators.

    Args:
        deterministic_evaluators: List of deterministic evaluator names (None = all)
        llm_evaluators: List of LLM evaluator names (None = all)

    Returns:
        Configured EvaluationPipeline instance
    """
    return EvaluationPipeline(deterministic_evaluators, llm_evaluators)


def create_fast_evaluator() -> EvaluationPipeline:
    """
    Create fast evaluation pipeline with deterministic evaluators only.

    Returns:
        EvaluationPipeline with deterministic evaluators only
    """
    return EvaluationPipeline(
        deterministic_evaluators=EvaluatorRegistry.get_available_deterministic(),
        llm_evaluators=[]
    )


def create_thorough_evaluator() -> EvaluationPipeline:
    """
    Create thorough evaluation pipeline with LLM evaluators only.

    Returns:
        EvaluationPipeline with LLM evaluators only
    """
    return EvaluationPipeline(
        deterministic_evaluators=[],
        llm_evaluators=EvaluatorRegistry.get_available_llm()
    )
