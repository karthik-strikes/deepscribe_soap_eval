"""Simple SOAP note generation with dual engine support (DSPy and Direct LLM)."""

import dspy
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.exceptions import SOAPGenerationError


# ==================== BASE INTERFACE ====================

class SOAPEngine(ABC):
    @abstractmethod
    def generate_soap(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        pass


# ==================== DSPY ENGINE ====================

class ExtractSubjectiveInfo(dspy.Signature):
    patient_convo: str = dspy.InputField(
        desc="Patient-provider conversation transcript")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background")
    chief_complaint: str = dspy.OutputField(
        desc="Primary reason for the visit")
    history_present_illness: str = dspy.OutputField(
        desc="Detailed account of current illness")
    review_of_systems: str = dspy.OutputField(desc="Review of body systems")
    past_medical_history: str = dspy.OutputField(
        desc="Past medical conditions")
    medications: str = dspy.OutputField(desc="Current medications")
    allergies: str = dspy.OutputField(desc="Known allergies")
    social_history: str = dspy.OutputField(desc="Lifestyle factors")
    family_history: str = dspy.OutputField(desc="Family medical history")


class ExtractObjectiveInfo(dspy.Signature):
    patient_convo: str = dspy.InputField(
        desc="Patient-provider conversation transcript")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background")
    vital_signs: str = dspy.OutputField(desc="Vital signs")
    physical_examination: str = dspy.OutputField(desc="Physical exam findings")
    diagnostic_tests: str = dspy.OutputField(desc="Lab and imaging results")
    mental_status: str = dspy.OutputField(desc="Mental status exam")


class GenerateAssessment(dspy.Signature):
    subjective_findings: str = dspy.InputField(desc="Subjective information")
    objective_findings: str = dspy.InputField(desc="Objective findings")
    patient_metadata: str = dspy.InputField(desc="Patient demographics")
    primary_diagnosis: str = dspy.OutputField(desc="Primary diagnosis")
    differential_diagnoses: str = dspy.OutputField(
        desc="Alternative diagnoses")
    problem_list: str = dspy.OutputField(desc="Active problems")
    clinical_reasoning: str = dspy.OutputField(desc="Clinical reasoning")


class DevelopPlan(dspy.Signature):
    assessment: str = dspy.InputField(desc="Clinical assessment")
    subjective_findings: str = dspy.InputField(desc="Patient information")
    objective_findings: str = dspy.InputField(desc="Clinical findings")
    immediate_treatment: str = dspy.OutputField(desc="Immediate treatments")
    medications_prescribed: str = dspy.OutputField(desc="Medications")
    diagnostic_orders: str = dspy.OutputField(desc="Additional tests")
    follow_up_instructions: str = dspy.OutputField(desc="Follow-up plan")
    patient_education: str = dspy.OutputField(desc="Patient education")
    precautions: str = dspy.OutputField(desc="Warning signs")


class CompileSOAPNote(dspy.Signature):
    subjective_section: str = dspy.InputField(desc="Subjective section")
    objective_section: str = dspy.InputField(desc="Objective section")
    assessment_section: str = dspy.InputField(desc="Assessment section")
    plan_section: str = dspy.InputField(desc="Plan section")
    patient_metadata: str = dspy.InputField(desc="Patient demographics")
    complete_soap_note: str = dspy.OutputField(desc="Final SOAP note")


class DSPySOAPEngine(SOAPEngine):
    def __init__(self):
        self.extract_subjective = dspy.Predict(ExtractSubjectiveInfo)
        self.extract_objective = dspy.Predict(ExtractObjectiveInfo)
        self.generate_assessment = dspy.Predict(GenerateAssessment)
        self.develop_plan = dspy.Predict(DevelopPlan)
        self.compile_soap = dspy.Predict(CompileSOAPNote)
        self.engine_type = "dspy"
        self.version = "1.0"

    def _prediction_to_dict(self, prediction):
        """Convert DSPy Prediction object to JSON-serializable dictionary"""
        if hasattr(prediction, '__dict__'):
            # Convert prediction object to dict, excluding private attributes
            result = {}
            for key, value in prediction.__dict__.items():
                if not key.startswith('_'):
                    result[key] = value
            return result
        return prediction

    def generate_soap(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        try:
            # Extract subjective
            subjective_result = self.extract_subjective(
                patient_convo=patient_convo, patient_metadata=patient_metadata
            )
            subjective_section = f"""SUBJECTIVE:
Chief Complaint: {subjective_result.chief_complaint}
History of Present Illness: {subjective_result.history_present_illness}
Review of Systems: {subjective_result.review_of_systems}
Past Medical History: {subjective_result.past_medical_history}
Medications: {subjective_result.medications}
Allergies: {subjective_result.allergies}
Social History: {subjective_result.social_history}
Family History: {subjective_result.family_history}"""

            # Extract objective
            objective_result = self.extract_objective(
                patient_convo=patient_convo, patient_metadata=patient_metadata
            )
            objective_section = f"""OBJECTIVE:
Vital Signs: {objective_result.vital_signs}
Physical Examination: {objective_result.physical_examination}
Diagnostic Tests: {objective_result.diagnostic_tests}
Mental Status: {objective_result.mental_status}"""

            # Generate assessment
            assessment_result = self.generate_assessment(
                subjective_findings=subjective_section,
                objective_findings=objective_section,
                patient_metadata=patient_metadata
            )
            assessment_section = f"""ASSESSMENT:
Primary Diagnosis: {assessment_result.primary_diagnosis}
Differential Diagnoses: {assessment_result.differential_diagnoses}
Problem List: {assessment_result.problem_list}
Clinical Reasoning: {assessment_result.clinical_reasoning}"""

            # Develop plan
            plan_result = self.develop_plan(
                assessment=assessment_section,
                subjective_findings=subjective_section,
                objective_findings=objective_section
            )
            plan_section = f"""PLAN:
Immediate Treatment: {plan_result.immediate_treatment}
Medications: {plan_result.medications_prescribed}
Diagnostic Orders: {plan_result.diagnostic_orders}
Follow-up: {plan_result.follow_up_instructions}
Patient Education: {plan_result.patient_education}
Precautions: {plan_result.precautions}"""

            # Compile final note
            final_soap = self.compile_soap(
                subjective_section=subjective_section,
                objective_section=objective_section,
                assessment_section=assessment_section,
                plan_section=plan_section,
                patient_metadata=patient_metadata
            )

            return {
                'complete_soap_note': final_soap.complete_soap_note,
                'subjective_components': self._prediction_to_dict(subjective_result),
                'objective_components': self._prediction_to_dict(objective_result),
                'assessment_components': self._prediction_to_dict(assessment_result),
                'plan_components': self._prediction_to_dict(plan_result),
                'engine_type': self.engine_type,
                'engine_info': {
                    'engine_type': self.engine_type,
                    'version': self.version,
                    'framework': 'DSPy'
                }
            }

        except Exception as e:
            raise SOAPGenerationError(f"DSPy SOAP generation failed: {e}")


# ==================== LLM ENGINE ====================

class LLMSOAPEngine(SOAPEngine):
    def __init__(self, llm_client, model_name: str = "gpt-4"):
        self.llm_client = llm_client
        self.model_name = model_name
        self.engine_type = "llm"
        self.version = "1.0"

    def _call_llm(self, prompt: str) -> str:
        try:
            if hasattr(self.llm_client, 'chat'):  # OpenAI style
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            else:  # Generic client
                return self.llm_client.generate(prompt, model=self.model_name)
        except Exception as e:
            raise SOAPGenerationError(f"LLM call failed: {e}")

    def generate_soap(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        try:
            prompt = f"""You are a medical doctor. Generate a complete SOAP note from this patient conversation.

Patient Metadata: {patient_metadata}

Conversation:
{patient_convo}

Format your response as a complete SOAP note with these sections:
- SUBJECTIVE (Chief Complaint, HPI, ROS, PMH, Medications, Allergies, Social History, Family History)
- OBJECTIVE (Vital Signs, Physical Exam, Diagnostic Tests, Mental Status)
- ASSESSMENT (Primary Diagnosis, Differential Diagnoses, Problem List, Clinical Reasoning)
- PLAN (Immediate Treatment, Medications, Diagnostic Orders, Follow-up, Patient Education, Precautions)

Provide a well-structured, professional medical note."""

            soap_note = self._call_llm(prompt)

            return {
                'complete_soap_note': soap_note,
                'subjective_components': None,  # LLM generates complete note
                'objective_components': None,
                'assessment_components': None,
                'plan_components': None,
                'engine_type': self.engine_type,
                'engine_info': {
                    'engine_type': self.engine_type,
                    'version': self.version,
                    'framework': 'Direct LLM',
                    'model': self.model_name
                }
            }

        except Exception as e:
            raise SOAPGenerationError(f"LLM SOAP generation failed: {e}")


# ==================== MAIN PIPELINE ====================

class SOAPGenerationPipeline:
    def __init__(self, engine_type: str = "dspy", **engine_kwargs):
        if engine_type == "dspy":
            self.engine = DSPySOAPEngine()
        elif engine_type == "llm":
            self.engine = LLMSOAPEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        self.engine_type = engine_type

    def forward(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        result = self.engine.generate_soap(patient_convo, patient_metadata)
        # Add pipeline-level metadata
        result['pipeline_info'] = {
            'pipeline_version': '2.0',
            'engine_used': self.engine_type
        }
        return result

    def switch_engine(self, engine_type: str, **engine_kwargs):
        """Switch to different engine"""
        if engine_type == "dspy":
            self.engine = DSPySOAPEngine()
        elif engine_type == "llm":
            self.engine = LLMSOAPEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        self.engine_type = engine_type
