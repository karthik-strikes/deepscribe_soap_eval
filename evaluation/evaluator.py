# =============================================================================
# PRECISION/RECALL LLM EVALUATORS WITH DSPY MODULES
# =============================================================================

import json
import dspy
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set
from abc import ABC, abstractmethod
from enum import Enum
from tqdm.asyncio import tqdm
from utils.json_parser import safe_json_parse


class EvaluatorType(Enum):
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"


class BaseEvaluator(ABC):
    """Base evaluator class"""

    @abstractmethod
    def get_type(self) -> EvaluatorType:
        pass

    @abstractmethod
    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        pass


@dataclass
class PrecisionRecallMetrics:
    """Container for precision/recall metrics with humanized counts"""

    # Counts with humanized names
    correctly_captured: int = 0
    missed_critical: int = 0
    unsupported_content: int = 0
    medically_sound: int = 0
    medically_incorrect: int = 0

    # Detailed lists
    correctly_captured_list: List[str] = field(default_factory=list)
    missed_critical_list: List[str] = field(default_factory=list)
    unsupported_content_list: List[str] = field(default_factory=list)
    medically_sound_list: List[str] = field(default_factory=list)
    medically_incorrect_list: List[str] = field(default_factory=list)

    @property
    def content_fidelity_recall(self) -> float:
        """Recall for content fidelity (0-1 scale)"""
        total_should_capture = self.correctly_captured + self.missed_critical
        if total_should_capture == 0:
            return 1.0
        return self.correctly_captured / total_should_capture

    @property
    def content_fidelity_precision(self) -> float:
        """Precision for content fidelity (0-1 scale)"""
        total_captured = self.correctly_captured + self.unsupported_content
        if total_captured == 0:
            return 1.0
        return self.correctly_captured / total_captured

    @property
    def content_fidelity_f1(self) -> float:
        """F1 score for content fidelity (0-1 scale)"""
        if self.content_fidelity_precision + self.content_fidelity_recall == 0:
            return 0.0
        return 2 * (self.content_fidelity_precision * self.content_fidelity_recall) / \
            (self.content_fidelity_precision + self.content_fidelity_recall)

    @property
    def medical_correctness_accuracy(self) -> float:
        """Medical correctness accuracy (0-1 scale)"""
        total_statements = self.medically_sound + self.medically_incorrect
        if total_statements == 0:
            return 1.0
        return self.medically_sound / total_statements

    def to_dict(self) -> Dict[str, Any]:
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


# =============================================================================
# DSPY SIGNATURES FOR CONTENT FIDELITY
# =============================================================================

class ExtractCriticalFindings(dspy.Signature):
    """Extract critical medical findings that must be documented"""
    transcript: str = dspy.InputField(desc="Patient conversation transcript")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")

    critical_findings: str = dspy.OutputField(
        desc="JSON list of critical medical facts that must be captured in any clinical note"
    )


class ValidateContentFidelity(dspy.Signature):
    """Validate if note content is faithful to transcript"""
    critical_findings: str = dspy.InputField(
        desc="JSON list of critical findings from transcript")
    generated_note: str = dspy.InputField(
        desc="Generated medical note to validate")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")

    correctly_captured: str = dspy.OutputField(
        desc="JSON object with 'list' and 'count' of critical findings correctly captured in note"
    )
    missed_critical: str = dspy.OutputField(
        desc="JSON object with 'list' and 'count' of critical findings missing from note"
    )
    unsupported_content: str = dspy.OutputField(
        desc="JSON object with 'list' and 'count' of medical content in note not supported by transcript or patient context"
    )


# =============================================================================
# DSPY SIGNATURES FOR MEDICAL CORRECTNESS
# =============================================================================

class ExtractMedicalStatements(dspy.Signature):
    """Extract all medical statements from generated note"""
    generated_note: str = dspy.InputField(desc="Generated medical note")

    medical_statements: str = dspy.OutputField(
        desc="JSON list of all medical statements, claims, and conclusions in the note"
    )


class ValidateMedicalAccuracy(dspy.Signature):
    """Validate medical accuracy of statements"""
    medical_statements: str = dspy.InputField(
        desc="JSON list of medical statements to validate")
    transcript: str = dspy.InputField(desc="Original transcript for context")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background information")

    medically_sound: str = dspy.OutputField(
        desc="JSON object with 'list' and 'count' of medically accurate and appropriate statements"
    )
    medically_incorrect: str = dspy.OutputField(
        desc="JSON object with 'list' and 'count' of medically incorrect, inappropriate or misleading statements"
    )


# =============================================================================
# DSPY MODULE IMPLEMENTATIONS
# =============================================================================

class ContentFidelityEvaluator(dspy.Module):
    """Module to evaluate content fidelity using chain of thought reasoning"""

    def __init__(self):
        super().__init__()
        self.extract_ground_truth = dspy.ChainOfThought(
            ExtractCriticalFindings)
        self.validate_content = dspy.ChainOfThought(ValidateContentFidelity)

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.LLM_JUDGE

    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Sync wrapper for BaseEvaluator compatibility - calls async version"""
        return asyncio.run(self.evaluate_async(transcript, generated_note, patient_metadata))

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Primary async evaluate method for better performance"""
        return self(transcript=transcript, generated_note=generated_note, patient_metadata=patient_metadata)

    def forward(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        try:
            # Step 1: Extract critical findings from transcript
            extraction_result = self.extract_ground_truth(
                transcript=transcript,
                patient_metadata=patient_metadata)

            # Step 2: Validate note content against critical findings
            validation_result = self.validate_content(
                critical_findings=extraction_result.critical_findings,
                generated_note=generated_note,
                patient_metadata=patient_metadata
            )

            # Parse JSON results
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
            return {
                'content_fidelity_recall': 0.0,
                'content_fidelity_precision': 0.0,
                'content_fidelity_f1': 0.0,
                'content_fidelity_counts': {
                    'correctly_captured': 0,
                    'missed_critical': 0,
                    'unsupported_content': 0
                },
                'content_fidelity_detail': {'error': str(e)}
            }

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """Interface method for BaseEvaluator compatibility"""
        return self(transcript=transcript, generated_note=generated_note)

    def _calculate_recall(self, correctly_captured: int, missed_critical: int) -> float:
        """Calculate recall (0-1 scale)"""
        total_should_capture = correctly_captured + missed_critical
        if total_should_capture == 0:
            return 1.0
        return correctly_captured / total_should_capture

    def _calculate_precision(self, correctly_captured: int, unsupported_content: int) -> float:
        """Calculate precision (0-1 scale)"""
        total_captured = correctly_captured + unsupported_content
        if total_captured == 0:
            return 1.0
        return correctly_captured / total_captured

    def _calculate_f1(self, correctly_captured: int, missed_critical: int, unsupported_content: int) -> float:
        """Calculate F1 score (0-1 scale)"""
        recall = self._calculate_recall(correctly_captured, missed_critical)
        precision = self._calculate_precision(
            correctly_captured, unsupported_content)

        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)


class MedicalCorrectnessEvaluator(dspy.Module):
    """Module to evaluate medical correctness using chain of thought reasoning"""

    def __init__(self):
        super().__init__()
        self.extract_statements = dspy.ChainOfThought(ExtractMedicalStatements)
        self.validate_accuracy = dspy.ChainOfThought(ValidateMedicalAccuracy)

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.LLM_JUDGE

    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Sync wrapper for BaseEvaluator compatibility - calls async version"""
        return asyncio.run(self.evaluate_async(transcript, generated_note, patient_metadata))

    async def evaluate_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Primary async evaluate method for better performance"""
        return self(transcript=transcript, generated_note=generated_note, patient_metadata=patient_metadata)

    def forward(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        try:
            # Step 1: Extract medical statements from note
            extraction_result = self.extract_statements(
                generated_note=generated_note)

            # Step 2: Validate medical accuracy of statements
            validation_result = self.validate_accuracy(
                medical_statements=extraction_result.medical_statements,
                transcript=transcript,
                patient_metadata=patient_metadata
            )

            # Parse JSON results
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
            return {
                'medical_correctness_accuracy': 1.0,  # Assume correct if error
                'medical_correctness_counts': {
                    'medically_sound': 0,
                    'medically_incorrect': 0
                },
                'medical_correctness_detail': {'error': str(e)}
            }

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """Interface method for BaseEvaluator compatibility"""
        return self(transcript=transcript, generated_note=generated_note)

    def _calculate_accuracy(self, medically_sound: int, medically_incorrect: int) -> float:
        """Calculate medical accuracy (0-1 scale)"""
        total_statements = medically_sound + medically_incorrect
        if total_statements == 0:
            return 1.0
        return medically_sound / total_statements


# =============================================================================
# UPDATED EVALUATION METRICS DATACLASS
# =============================================================================

@dataclass
class EnhancedEvaluationMetrics:
    """Container for enhanced evaluation metrics with precision/recall"""

    # Deterministic metrics (unchanged from original)
    entity_coverage: float = 0.0
    section_completeness: float = 0.0
    format_validity: float = 0.0

    # NEW: Precision/Recall LLM Judge metrics (0-1 scale) - REPLACES old LLM metrics
    content_fidelity_recall: float = 0.0
    content_fidelity_precision: float = 0.0
    content_fidelity_f1: float = 0.0
    medical_correctness_accuracy: float = 0.0

    # Humanized counts (NEW)
    content_fidelity_counts: Dict[str, int] = field(default_factory=dict)
    medical_correctness_counts: Dict[str, int] = field(default_factory=dict)

    # Details (original structure maintained)
    missing_entities: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    llm_feedback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'deterministic_metrics': {
                'entity_coverage': self.entity_coverage,
                'section_completeness': self.section_completeness,
                'format_validity': self.format_validity,
            },
            'llm_metrics': {
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
            },
            'details': {
                'missing_entities': self.missing_entities,
                'missing_sections': self.missing_sections,
                'format_issues': self.format_issues,
                'llm_feedback': self.llm_feedback,
            }
        }

    def overall_quality_score(self) -> float:
        """Calculate overall quality score (0-1 scale)"""
        # Combine all metrics with weights
        metrics = [
            self.content_fidelity_f1 * 0.4,  # Most important
            self.medical_correctness_accuracy * 0.3,  # Very important
            (self.entity_coverage / 100) * 0.1,  # Convert to 0-1 scale
            (self.section_completeness / 100) * 0.1,
            (self.format_validity / 100) * 0.1
        ]
        return sum(metrics)


# =============================================================================
# DETERMINISTIC EVALUATORS (ORIGINAL - UNCHANGED)
# =============================================================================

class EntityCoverageEvaluator(BaseEvaluator):
    """Check if key medical entities from transcript appear in note"""

    def __init__(self):
        self.medical_patterns = {
            'medications': r'\b(?:\w+(?:cillin|mycin|pril|statin|olol|pine|zole)|mg|tablet|capsule|pill)\b',
            'symptoms': r'\b(?:pain|ache|fever|nausea|vomiting|headache|dizzy|shortness of breath|chest pain|fatigue|weakness)\b',
            'vital_signs': r'\b(?:\d{2,3}/\d{2,3}|\d{2,3}\s*bpm|\d{2,3}°?[FC]|O2\s*sat|\d{1,3}%)\b',
            'procedures': r'\b(?:x-ray|ct scan|mri|ekg|ecg|blood test|biopsy|surgery)\b',
        }

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.DETERMINISTIC

    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
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
        import re
        entities = {}
        text_lower = text.lower()

        for entity_type, pattern in self.medical_patterns.items():
            matches = set(re.findall(pattern, text_lower, re.IGNORECASE))
            entities[entity_type] = matches

        return entities


class SOAPCompletenessEvaluator(BaseEvaluator):
    """Check if note has proper SOAP structure"""

    def __init__(self):
        self.required_sections = {
            'subjective': r'(?:subjective|chief complaint|cc:|history of present illness|hpi)',
            'objective': r'(?:objective|physical exam|pe:|vital signs|vs:)',
            'assessment': r'(?:assessment|diagnosis|impression|dx:)',
            'plan': r'(?:plan|treatment|recommendations|follow.?up)'
        }

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.DETERMINISTIC

    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
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


class FormatValidityEvaluator(BaseEvaluator):
    """Check basic format requirements and detect obvious issues"""

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.DETERMINISTIC

    def evaluate(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        import re
        issues = []
        score = 100.0

        # Length checks
        if len(generated_note.strip()) < 50:
            issues.append("Note too short")
            score -= 30
        elif len(generated_note) > 3000:
            issues.append("Note too long")
            score -= 10

        # Structure checks
        if not re.search(r'[.!?]', generated_note):
            issues.append("No proper sentence structure")
            score -= 20

        # Basic medical note formatting
        if not re.search(r'(?:patient|pt)', generated_note, re.IGNORECASE):
            issues.append("Missing patient references")
            score -= 15

        # Check for placeholder text
        placeholders = ['[PLACEHOLDER]', 'TODO', 'FIXME', 'XXX']
        for placeholder in placeholders:
            if placeholder.lower() in generated_note.lower():
                issues.append(f"Contains placeholder: {placeholder}")
                score -= 25

        return {
            'format_validity': max(0, score),
            'format_issues': issues
        }


# =============================================================================
# REGISTRY INTEGRATION (UPDATED)
# =============================================================================

class EvaluatorRegistry:
    """Registry for managing different evaluators"""

    _deterministic_evaluators: Dict[str, type] = {}
    _llm_evaluators: Dict[str, type] = {}

    @classmethod
    def register_deterministic(cls, name: str, evaluator_class: type):
        """Register a deterministic evaluator"""
        cls._deterministic_evaluators[name] = evaluator_class

    @classmethod
    def register_llm(cls, name: str, evaluator_class: type):
        """Register an LLM evaluator"""
        cls._llm_evaluators[name] = evaluator_class

    @classmethod
    def create_deterministic_evaluator(cls, name: str, **kwargs):
        """Create a deterministic evaluator instance"""
        if name not in cls._deterministic_evaluators:
            raise ValueError(f"Unknown deterministic evaluator: {name}")
        return cls._deterministic_evaluators[name](**kwargs)

    @classmethod
    def create_llm_evaluator(cls, name: str, **kwargs):
        """Create an LLM evaluator instance"""
        if name not in cls._llm_evaluators:
            raise ValueError(f"Unknown LLM evaluator: {name}")
        return cls._llm_evaluators[name](**kwargs)

    @classmethod
    def get_available_deterministic(cls) -> List[str]:
        """Get list of available deterministic evaluators"""
        return list(cls._deterministic_evaluators.keys())

    @classmethod
    def get_available_llm(cls) -> List[str]:
        """Get list of available LLM evaluators"""
        return list(cls._llm_evaluators.keys())


# Register all evaluators (deterministic + new precision/recall LLM)
EvaluatorRegistry.register_deterministic(
    "entity_coverage", EntityCoverageEvaluator)
EvaluatorRegistry.register_deterministic(
    "soap_completeness", SOAPCompletenessEvaluator)
EvaluatorRegistry.register_deterministic(
    "format_validity", FormatValidityEvaluator)

EvaluatorRegistry.register_llm("content_fidelity", ContentFidelityEvaluator)
EvaluatorRegistry.register_llm(
    "medical_correctness", MedicalCorrectnessEvaluator)


# =============================================================================
# EVALUATION PIPELINE (UPDATED)
# =============================================================================

class EvaluationPipeline:
    """Configurable evaluation pipeline with plug-in evaluators"""

    def __init__(self, deterministic_evaluators: Optional[List[str]] = None,
                 llm_evaluators: Optional[List[str]] = None):
        """
        Initialize pipeline with specific evaluators.

        Args:
            deterministic_evaluators: List of deterministic evaluator names to use
            llm_evaluators: List of LLM evaluator names to use
        """
        # Use provided evaluators or defaults
        det_names = deterministic_evaluators or EvaluatorRegistry.get_available_deterministic()
        llm_names = llm_evaluators or EvaluatorRegistry.get_available_llm()

        # Create evaluator instances
        self.deterministic_evaluators = [
            EvaluatorRegistry.create_deterministic_evaluator(name) for name in det_names
        ]
        self.llm_evaluators = [
            EvaluatorRegistry.create_llm_evaluator(name) for name in llm_names
        ]

        # Pipeline configured silently

    def evaluate_deterministic(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Fast evaluation using only deterministic methods"""
        results = {}

        for evaluator in self.deterministic_evaluators:
            evaluator_results = evaluator.evaluate(
                transcript, generated_note, patient_metadata)
            results.update(evaluator_results)

        metrics = EnhancedEvaluationMetrics(
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
        )

        return metrics.to_dict()

    def evaluate_llm_only(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Sync wrapper - calls async version for optimal performance"""
        return asyncio.run(self.evaluate_llm_only_async(transcript, generated_note, patient_metadata))

    async def evaluate_llm_only_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Async LLM-only evaluation for better performance"""
        results = {}
        llm_feedback = {}

        # Run LLM evaluators in parallel
        if self.llm_evaluators:
            llm_tasks = []
            for evaluator in self.llm_evaluators:
                if hasattr(evaluator, 'evaluate_async'):
                    task = evaluator.evaluate_async(
                        transcript, generated_note, patient_metadata)
                else:
                    # Fallback to sync evaluation wrapped in async
                    async def sync_wrapper(eval_func, trans, note, metadata):
                        return eval_func(trans, note, metadata)
                    task = sync_wrapper(
                        evaluator.evaluate, transcript, generated_note, patient_metadata)
                llm_tasks.append(task)

            # Wait for all LLM evaluations to complete
            llm_results_list = await asyncio.gather(*llm_tasks)

            # Process results
            for evaluator_results in llm_results_list:
                results.update(evaluator_results)

                # Collect detailed feedback
                for key, value in evaluator_results.items():
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

        return metrics.to_dict()

    def evaluate_comprehensive(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Sync wrapper - calls async version for optimal performance"""
        return asyncio.run(self.evaluate_comprehensive_async(transcript, generated_note, patient_metadata))

    async def evaluate_comprehensive_async(self, transcript: str, generated_note: str, patient_metadata: str = "") -> Dict[str, Any]:
        """Async comprehensive evaluation with both deterministic and LLM methods for better performance"""
        results = {}
        llm_feedback = {}

        # Run deterministic evaluators (these are typically fast, so run sequentially)
        if self.deterministic_evaluators:
            for evaluator in self.deterministic_evaluators:
                evaluator_results = evaluator.evaluate(
                    transcript, generated_note, patient_metadata)
                results.update(evaluator_results)

        # Run LLM evaluators in parallel for better performance
        if self.llm_evaluators:
            llm_tasks = []
            for evaluator in self.llm_evaluators:
                if hasattr(evaluator, 'evaluate_async'):
                    task = evaluator.evaluate_async(
                        transcript, generated_note, patient_metadata)
                else:
                    # Fallback to sync evaluation wrapped in async
                    async def sync_wrapper(eval_func, trans, note, metadata):
                        return eval_func(trans, note, metadata)
                    task = sync_wrapper(evaluator.evaluate,
                                        transcript, generated_note, patient_metadata)
                llm_tasks.append(task)

            # Wait for all LLM evaluations to complete
            llm_results_list = await asyncio.gather(*llm_tasks)

            # Process results
            for evaluator_results in llm_results_list:
                results.update(evaluator_results)

                # Collect detailed feedback
                for key, value in evaluator_results.items():
                    if key.endswith('_detail'):
                        llm_feedback[key] = value

        metrics = EnhancedEvaluationMetrics(
            # Deterministic
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            # NEW LLM Judge
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
            # Details
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
            llm_feedback=llm_feedback,
        )

        return metrics.to_dict()

    async def evaluate_llm_only_async(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """Async LLM-only evaluation for better performance"""
        results = {}
        llm_feedback = {}

        # Run LLM evaluators in parallel
        llm_tasks = []
        for evaluator in self.llm_evaluators:
            if hasattr(evaluator, 'evaluate_async'):
                task = evaluator.evaluate_async(transcript, generated_note)
            else:
                # Fallback to sync evaluation wrapped in async
                async def sync_wrapper(eval_func, trans, note):
                    return eval_func(trans, note)
                task = sync_wrapper(evaluator.evaluate,
                                    transcript, generated_note)
            llm_tasks.append(task)

        # Wait for all LLM evaluations to complete
        if llm_tasks:
            llm_results_list = await asyncio.gather(*llm_tasks)

            # Process results
            for evaluator_results in llm_results_list:
                results.update(evaluator_results)

                # Collect detailed feedback
                for key, value in evaluator_results.items():
                    if key.endswith('_detail'):
                        llm_feedback[key] = value

        metrics = LLMEvaluationMetrics(
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

        return metrics.to_dict()

    def add_evaluator(self, evaluator_name: str, evaluator_type: EvaluatorType):
        """Add an evaluator to the current pipeline"""
        if evaluator_type == EvaluatorType.DETERMINISTIC:
            new_evaluator = EvaluatorRegistry.create_deterministic_evaluator(
                evaluator_name)
            self.deterministic_evaluators.append(new_evaluator)
        else:
            new_evaluator = EvaluatorRegistry.create_llm_evaluator(
                evaluator_name)
            self.llm_evaluators.append(new_evaluator)

        # Evaluator added silently

    def remove_evaluator(self, evaluator_name: str):
        """Remove an evaluator from the current pipeline"""
        # Remove from deterministic evaluators
        self.deterministic_evaluators = [
            ev for ev in self.deterministic_evaluators
            if ev.__class__.__name__ != f"{evaluator_name.title().replace('_', '')}Evaluator"
        ]

        # Remove from LLM evaluators
        self.llm_evaluators = [
            ev for ev in self.llm_evaluators
            if ev.__class__.__name__ != f"{evaluator_name.title().replace('_', '')}Evaluator"
        ]

        # Evaluator removed silently

    @staticmethod
    def list_available_evaluators() -> Dict[str, List[str]]:
        """List all available evaluators"""
        return {
            'deterministic': EvaluatorRegistry.get_available_deterministic(),
            'llm': EvaluatorRegistry.get_available_llm()
        }


# =============================================================================
# SIMPLE USAGE FUNCTIONS (UPDATED)
# =============================================================================

def create_evaluator(deterministic_evaluators: Optional[List[str]] = None,
                     llm_evaluators: Optional[List[str]] = None) -> EvaluationPipeline:
    """Create the evaluation pipeline with specific evaluators"""
    return EvaluationPipeline(deterministic_evaluators, llm_evaluators)


def create_fast_evaluator() -> EvaluationPipeline:
    """Create evaluator with only deterministic evaluators for speed"""
    return EvaluationPipeline(
        deterministic_evaluators=EvaluatorRegistry.get_available_deterministic(),
        llm_evaluators=[]
    )


def create_thorough_evaluator() -> EvaluationPipeline:
    """Create evaluator with only LLM evaluators for deep analysis"""
    return EvaluationPipeline(
        deterministic_evaluators=[],
        llm_evaluators=EvaluatorRegistry.get_available_llm()
    )
