"""Simple SOAP note generation with dual engine support (DSPy and Direct LLM)."""

import dspy
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any
from core.exceptions import SOAPGenerationError


# ==================== BASE INTERFACE ====================

class SOAPEngine(ABC):
    @abstractmethod
    def generate_soap(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        pass

    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """Async version of generate_soap - default implementation calls sync version"""
        return self.generate_soap(patient_convo, patient_metadata)


# ==================== DSPY ENGINE ====================

class ExtractSubjectiveInfo(dspy.Signature):
    """Extract patient's subjective information from conversation. Handle all PHI with HIPAA compliance."""
    patient_convo: str = dspy.InputField(
        desc="Patient-provider conversation transcript")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background")
    subjective_section: str = dspy.OutputField(
        desc="Complete SUBJECTIVE section including chief complaint, HPI, ROS, PMH, medications, allergies, social and family history"
    )


class ExtractObjectiveInfo(dspy.Signature):
    """Extract objective clinical findings from conversation. Handle all PHI with HIPAA compliance."""
    patient_convo: str = dspy.InputField(
        desc="Patient-provider conversation transcript")
    patient_metadata: str = dspy.InputField(
        desc="Patient demographics and background")
    objective_section: str = dspy.OutputField(
        desc="Complete OBJECTIVE section including vital signs, physical exam, diagnostic tests, and mental status"
    )


class GenerateAssessmentAndPlan(dspy.Signature):
    """Generate clinical assessment and treatment plan. Handle all PHI with HIPAA compliance."""
    subjective_section: str = dspy.InputField(desc="Subjective findings")
    objective_section: str = dspy.InputField(desc="Objective findings")
    patient_metadata: str = dspy.InputField(desc="Patient demographics")
    assessment_section: str = dspy.OutputField(
        desc="Complete ASSESSMENT section with primary diagnosis, differential diagnoses, and clinical reasoning"
    )
    plan_section: str = dspy.OutputField(
        desc="Complete PLAN section with treatments, medications, orders, follow-up, education, and precautions"
    )


class DSPySOAPEngine(SOAPEngine):
    def __init__(self):
        self.extract_subjective = dspy.ChainOfThought(ExtractSubjectiveInfo)
        self.extract_objective = dspy.ChainOfThought(ExtractObjectiveInfo)
        self.generate_assessment_plan = dspy.ChainOfThought(
            GenerateAssessmentAndPlan)
        self.engine_type = "dspy"
        self.version = "1.0"

    def generate_soap(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        try:
            # Step 1: Extract subjective information
            subjective_result = self.extract_subjective(
                patient_convo=patient_convo,
                patient_metadata=patient_metadata
            )

            # Step 2: Extract objective information
            objective_result = self.extract_objective(
                patient_convo=patient_convo,
                patient_metadata=patient_metadata
            )

            # Step 3: Generate assessment and plan based on findings
            assessment_plan_result = self.generate_assessment_plan(
                subjective_section=subjective_result.subjective_section,
                objective_section=objective_result.objective_section,
                patient_metadata=patient_metadata
            )

            return {
                'subjective': subjective_result.subjective_section,
                'objective': objective_result.objective_section,
                'assessment': assessment_plan_result.assessment_section,
                'plan': assessment_plan_result.plan_section,
                'engine_type': self.engine_type,
                'version': self.version
            }
        except Exception as e:
            raise SOAPGenerationError(f"SOAP generation failed: {e}")

    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """Async version with parallel DSPy operations for better performance"""
        try:
            # Run all DSPy operations in parallel for better performance
            subjective_task = asyncio.create_task(
                asyncio.to_thread(self.extract_subjective,
                                  patient_convo=patient_convo,
                                  patient_metadata=patient_metadata)
            )

            objective_task = asyncio.create_task(
                asyncio.to_thread(self.extract_objective,
                                  patient_convo=patient_convo,
                                  patient_metadata=patient_metadata)
            )

            # Wait for both extraction tasks to complete
            subjective_result, objective_result = await asyncio.gather(
                subjective_task, objective_task
            )

            # Generate assessment and plan based on findings (this depends on the previous results)
            assessment_plan_result = await asyncio.to_thread(
                self.generate_assessment_plan,
                subjective_section=subjective_result.subjective_section,
                objective_section=objective_result.objective_section,
                patient_metadata=patient_metadata
            )

            return {
                'subjective': subjective_result.subjective_section,
                'objective': objective_result.objective_section,
                'assessment': assessment_plan_result.assessment_section,
                'plan': assessment_plan_result.plan_section,
                'engine_type': self.engine_type,
                'version': self.version
            }
        except Exception as e:
            raise SOAPGenerationError(f"Async SOAP generation failed: {e}")


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

    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """Async version of LLM SOAP generation"""
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

            # Run LLM call in thread to avoid blocking
            soap_note = await asyncio.to_thread(self._call_llm, prompt)

            return {
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
            raise SOAPGenerationError(f"Async LLM SOAP generation failed: {e}")


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
        """Sync wrapper - calls async version for optimal performance"""
        return asyncio.run(self.forward_async(patient_convo, patient_metadata))

    async def forward_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """Async version of forward for better performance"""
        result = await self.engine.generate_soap_async(patient_convo, patient_metadata)
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
