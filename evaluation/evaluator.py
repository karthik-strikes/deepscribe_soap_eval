"""
DeepScribe Streamlined Dual Evaluator with Registry System
=========================================================

Focused evaluation system with clear separation and plug-in architecture:
- Deterministic: Fast, objective metrics that can be computed reliably
- LLM Judge: Complex subjective assessments requiring medical reasoning
"""

import re
import dspy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Any, Type
from enum import Enum


class EvaluatorType(Enum):
    DETERMINISTIC = "deterministic"
    LLM_JUDGE = "llm_judge"


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    # Deterministic metrics - fast, objective
    entity_coverage: float = 0.0
    section_completeness: float = 0.0
    format_validity: float = 0.0

    # LLM Judge metrics - complex, subjective
    clinical_accuracy: float = 0.0
    missing_findings: float = 0.0
    hallucination_score: float = 0.0

    # Details
    missing_entities: List[str] = field(default_factory=list)
    missing_sections: List[str] = field(default_factory=list)
    format_issues: List[str] = field(default_factory=list)
    llm_feedback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'deterministic': {
                'entity_coverage': self.entity_coverage,
                'section_completeness': self.section_completeness,
                'format_validity': self.format_validity,
            },
            'llm_judge': {
                'clinical_accuracy': self.clinical_accuracy,
                'missing_findings': self.missing_findings,
                'hallucination_score': self.hallucination_score,
            },
            'details': {
                'missing_entities': self.missing_entities,
                'missing_sections': self.missing_sections,
                'format_issues': self.format_issues,
                'llm_feedback': self.llm_feedback,
            }
        }

    def overall_score(self) -> float:
        """Weighted composite score"""
        # Deterministic scores
        det_scores = [s for s in [self.entity_coverage,
                                  self.section_completeness, self.format_validity] if s > 0]
        det_score = sum(det_scores) / len(det_scores) if det_scores else 0

        # LLM scores
        llm_scores = [s for s in [self.clinical_accuracy,
                                  self.missing_findings, self.hallucination_score] if s > 0]
        llm_score = sum(llm_scores) / len(llm_scores) if llm_scores else 0

        # Weighted combination based on what's available
        if det_scores and llm_scores:
            return (det_score * 0.4 + llm_score * 0.6)  # Both available
        elif det_scores:
            return det_score  # Only deterministic
        elif llm_scores:
            return llm_score  # Only LLM
        else:
            return 0.0


class BaseEvaluator(ABC):
    """Base evaluator class"""

    @abstractmethod
    def get_type(self) -> EvaluatorType:
        pass

    @abstractmethod
    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        pass


# =============================================================================
# EVALUATOR REGISTRY - Plug-in system for evaluators
# =============================================================================

class EvaluatorRegistry:
    """Registry for managing different evaluators"""

    _deterministic_evaluators: Dict[str, Type[BaseEvaluator]] = {}
    _llm_evaluators: Dict[str, Type[BaseEvaluator]] = {}

    @classmethod
    def register_deterministic(cls, name: str, evaluator_class: Type[BaseEvaluator]):
        """Register a deterministic evaluator"""
        cls._deterministic_evaluators[name] = evaluator_class
        print(f"Registered deterministic evaluator: {name}")

    @classmethod
    def register_llm(cls, name: str, evaluator_class: Type[BaseEvaluator]):
        """Register an LLM evaluator"""
        cls._llm_evaluators[name] = evaluator_class
        print(f"Registered LLM evaluator: {name}")

    @classmethod
    def create_deterministic_evaluator(cls, name: str, **kwargs) -> BaseEvaluator:
        """Create a deterministic evaluator instance"""
        if name not in cls._deterministic_evaluators:
            raise ValueError(f"Unknown deterministic evaluator: {name}")
        return cls._deterministic_evaluators[name](**kwargs)

    @classmethod
    def create_llm_evaluator(cls, name: str, **kwargs) -> BaseEvaluator:
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

    @classmethod
    def create_all_deterministic(cls) -> List[BaseEvaluator]:
        """Create instances of all registered deterministic evaluators"""
        return [evaluator_class() for evaluator_class in cls._deterministic_evaluators.values()]

    @classmethod
    def create_all_llm(cls) -> List[BaseEvaluator]:
        """Create instances of all registered LLM evaluators"""
        return [evaluator_class() for evaluator_class in cls._llm_evaluators.values()]


# =============================================================================
# DETERMINISTIC EVALUATORS - Fast, objective metrics
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

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
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

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
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

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
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
# LLM JUDGE EVALUATORS - Complex medical reasoning
# =============================================================================

class ClinicalAccuracySignature(dspy.Signature):
    """Evaluate clinical accuracy and medical reasoning"""
    transcript: str = dspy.InputField(desc="Patient conversation transcript")
    generated_note: str = dspy.InputField(desc="Generated medical note")

    accuracy_score: float = dspy.OutputField(
        desc="Clinical accuracy score 0-100")
    issues: str = dspy.OutputField(
        desc="Specific clinical accuracy issues found")
    reasoning: str = dspy.OutputField(desc="Medical reasoning assessment")


class MissingCriticalFindingsSignature(dspy.Signature):
    """Identify missing critical medical findings"""
    transcript: str = dspy.InputField(desc="Patient conversation transcript")
    generated_note: str = dspy.InputField(desc="Generated medical note")

    missing_score: float = dspy.OutputField(
        desc="Score for missing findings (100 = nothing missing)")
    critical_missing: str = dspy.OutputField(
        desc="Critical findings that were missed")
    severity_assessment: str = dspy.OutputField(
        desc="Assessment of severity of omissions")


class HallucinationDetectionSignature(dspy.Signature):
    """Detect medical hallucinations and unsupported facts"""
    transcript: str = dspy.InputField(desc="Patient conversation transcript")
    generated_note: str = dspy.InputField(desc="Generated medical note")

    hallucination_score: float = dspy.OutputField(
        desc="Score for hallucinations (100 = no hallucinations)")
    hallucinated_content: str = dspy.OutputField(
        desc="Specific hallucinated medical content")
    confidence_level: str = dspy.OutputField(
        desc="Confidence in hallucination assessment")


class ClinicalAccuracyEvaluator(BaseEvaluator):
    """Evaluate clinical accuracy using medical knowledge"""

    def __init__(self):
        self.predictor = dspy.Predict(ClinicalAccuracySignature)

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.LLM_JUDGE

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        try:
            result = self.predictor(
                transcript=transcript, generated_note=generated_note)

            return {
                'clinical_accuracy': result.accuracy_score,
                'clinical_accuracy_detail': {
                    'issues': result.issues,
                    'reasoning': result.reasoning
                }
            }
        except Exception as e:
            return {
                'clinical_accuracy': 0.0,
                'clinical_accuracy_detail': {'error': str(e)}
            }


class MissingFindingsEvaluator(BaseEvaluator):
    """Detect missing critical findings"""

    def __init__(self):
        self.predictor = dspy.Predict(MissingCriticalFindingsSignature)

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.LLM_JUDGE

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        try:
            result = self.predictor(
                transcript=transcript, generated_note=generated_note)

            return {
                'missing_findings': result.missing_score,
                'missing_findings_detail': {
                    'critical_missing': result.critical_missing,
                    'severity_assessment': result.severity_assessment
                }
            }
        except Exception as e:
            return {
                'missing_findings': 100.0,  # Assume no missing if error
                'missing_findings_detail': {'error': str(e)}
            }


class HallucinationEvaluator(BaseEvaluator):
    """Detect hallucinated medical content"""

    def __init__(self):
        self.predictor = dspy.Predict(HallucinationDetectionSignature)

    def get_type(self) -> EvaluatorType:
        return EvaluatorType.LLM_JUDGE

    def evaluate(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        try:
            result = self.predictor(
                transcript=transcript, generated_note=generated_note)

            return {
                'hallucination_score': result.hallucination_score,
                'hallucination_detail': {
                    'hallucinated_content': result.hallucinated_content,
                    'confidence_level': result.confidence_level
                }
            }
        except Exception as e:
            return {
                'hallucination_score': 100.0,  # Assume no hallucination if error
                'hallucination_detail': {'error': str(e)}
            }


# Register default evaluators
EvaluatorRegistry.register_deterministic(
    "entity_coverage", EntityCoverageEvaluator)
EvaluatorRegistry.register_deterministic(
    "soap_completeness", SOAPCompletenessEvaluator)
EvaluatorRegistry.register_deterministic(
    "format_validity", FormatValidityEvaluator)

EvaluatorRegistry.register_llm("clinical_accuracy", ClinicalAccuracyEvaluator)
EvaluatorRegistry.register_llm("missing_findings", MissingFindingsEvaluator)
EvaluatorRegistry.register_llm("hallucination", HallucinationEvaluator)


# =============================================================================
# CONFIGURABLE EVALUATION PIPELINE
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

        print(f"Pipeline configured with {len(self.deterministic_evaluators)} deterministic "
              f"and {len(self.llm_evaluators)} LLM evaluators")

    def evaluate_deterministic(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """Fast evaluation using only deterministic methods"""
        results = {}

        for evaluator in self.deterministic_evaluators:
            evaluator_results = evaluator.evaluate(transcript, generated_note)
            results.update(evaluator_results)

        metrics = EvaluationMetrics(
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
        )

        return metrics.to_dict()

    def evaluate_llm_only(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """LLM-only evaluation for deep clinical assessment"""
        results = {}
        llm_feedback = {}

        # Run only LLM evaluators
        for evaluator in self.llm_evaluators:
            evaluator_results = evaluator.evaluate(transcript, generated_note)
            results.update(evaluator_results)

            # Collect detailed feedback
            for key, value in evaluator_results.items():
                if key.endswith('_detail'):
                    llm_feedback[key] = value

        metrics = EvaluationMetrics(
            # LLM Judge only
            clinical_accuracy=results.get('clinical_accuracy', 0.0),
            missing_findings=results.get('missing_findings', 0.0),
            hallucination_score=results.get('hallucination_score', 0.0),
            # Details
            llm_feedback=llm_feedback,
        )

        return metrics.to_dict()

    def evaluate_comprehensive(self, transcript: str, generated_note: str) -> Dict[str, Any]:
        """Comprehensive evaluation with both deterministic and LLM methods"""
        results = {}
        llm_feedback = {}

        # Run deterministic evaluators
        for evaluator in self.deterministic_evaluators:
            evaluator_results = evaluator.evaluate(transcript, generated_note)
            results.update(evaluator_results)

        # Run LLM evaluators
        for evaluator in self.llm_evaluators:
            evaluator_results = evaluator.evaluate(transcript, generated_note)
            results.update(evaluator_results)

            # Collect detailed feedback
            for key, value in evaluator_results.items():
                if key.endswith('_detail'):
                    llm_feedback[key] = value

        metrics = EvaluationMetrics(
            # Deterministic
            entity_coverage=results.get('entity_coverage', 0.0),
            section_completeness=results.get('section_completeness', 0.0),
            format_validity=results.get('format_validity', 0.0),
            # LLM Judge
            clinical_accuracy=results.get('clinical_accuracy', 0.0),
            missing_findings=results.get('missing_findings', 0.0),
            hallucination_score=results.get('hallucination_score', 0.0),
            # Details
            missing_entities=results.get('missing_entities', []),
            missing_sections=results.get('missing_sections', []),
            format_issues=results.get('format_issues', []),
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

        print(f"Added {evaluator_type.value} evaluator: {evaluator_name}")

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

        print(f"Removed evaluator: {evaluator_name}")

    @staticmethod
    def list_available_evaluators() -> Dict[str, List[str]]:
        """List all available evaluators"""
        return {
            'deterministic': EvaluatorRegistry.get_available_deterministic(),
            'llm': EvaluatorRegistry.get_available_llm()
        }


# =============================================================================
# SIMPLE USAGE FUNCTIONS
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
