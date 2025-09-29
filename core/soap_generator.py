"""
SOAP Note Generation with Dual Engine Support and True Async/Batch Processing
==============================================================================

Provides DSPy and LLM-based engines for generating medical SOAP notes with
proper async/await patterns and true batch processing capabilities.
"""

import dspy
import asyncio
import yaml
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from core.exceptions import SOAPGenerationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== BASE INTERFACE ====================

class SOAPEngine(ABC):
    """
    Abstract base class for SOAP note generation engines.

    All engines must implement both single and batch async generation methods.
    """

    @abstractmethod
    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """
        Generate a single SOAP note asynchronously.

        Args:
            patient_convo: Patient-provider conversation transcript
            patient_metadata: Patient demographics and background information

        Returns:
            Dictionary containing SOAP sections and metadata
        """
        pass

    @abstractmethod
    async def generate_soap_batch_async(self, conversations: List[str], metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple SOAP notes in batch.

        Args:
            conversations: List of conversation transcripts
            metadata_list: List of patient metadata (same length as conversations)

        Returns:
            List of dictionaries containing SOAP sections and metadata
        """
        pass


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
    """
    DSPy-based SOAP note generation engine.

    Uses Chain of Thought reasoning with parallel subjective/objective extraction
    for improved performance.
    """

    def __init__(self):
        """Initialize DSPy modules for SOAP generation."""
        self.extract_subjective = dspy.ChainOfThought(ExtractSubjectiveInfo)
        self.extract_objective = dspy.ChainOfThought(ExtractObjectiveInfo)
        self.generate_assessment_plan = dspy.ChainOfThought(
            GenerateAssessmentAndPlan)
        self.engine_type = "dspy"
        self.version = "1.0"

    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """
        Generate single SOAP note with parallel S/O extraction.

        Extracts Subjective and Objective sections in parallel for performance,
        then generates Assessment and Plan based on those findings.

        Args:
            patient_convo: Patient-provider conversation transcript
            patient_metadata: Patient demographics and background

        Returns:
            Dictionary with SOAP sections (subjective, objective, assessment, plan)

        Raises:
            SOAPGenerationError: If generation fails
        """
        try:
            # Create DSPy Examples for parallel extraction
            subjective_example = dspy.Example(
                patient_convo=patient_convo,
                patient_metadata=patient_metadata
            ).with_inputs("patient_convo", "patient_metadata")

            objective_example = dspy.Example(
                patient_convo=patient_convo,
                patient_metadata=patient_metadata
            ).with_inputs("patient_convo", "patient_metadata")

            # Parallel extraction using DSPy batch
            async def extract_parallel():
                """Run subjective and objective extractions in parallel."""
                subjective_task = asyncio.to_thread(
                    self.extract_subjective.batch,
                    examples=[subjective_example],
                    num_threads=1
                )
                objective_task = asyncio.to_thread(
                    self.extract_objective.batch,
                    examples=[objective_example],
                    num_threads=1
                )

                subj_results, obj_results = await asyncio.gather(subjective_task, objective_task)
                return subj_results[0], obj_results[0]

            subjective_result, objective_result = await extract_parallel()

            # Generate assessment and plan based on findings
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
            logger.error(f"DSPy async SOAP generation failed: {e}")
            raise SOAPGenerationError(f"Async SOAP generation failed: {e}")

    async def generate_soap_batch_async(self, conversations: List[str], metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple SOAP notes using true batch processing.

        Uses DSPy's native batch() method for efficient parallel processing.
        Handles failures gracefully and tracks original indices.

        Args:
            conversations: List of conversation transcripts
            metadata_list: List of patient metadata

        Returns:
            List of SOAP note dictionaries (includes 'original_index' for tracking)

        Raises:
            SOAPGenerationError: If batch generation fails
        """
        try:
            if len(conversations) != len(metadata_list):
                raise ValueError(
                    "Conversations and metadata lists must have same length")

            logger.info(
                f"Starting batch SOAP generation for {len(conversations)} conversations")

            # Create Examples for batch processing
            subjective_examples = [
                dspy.Example(patient_convo=conv, patient_metadata=meta).with_inputs(
                    "patient_convo", "patient_metadata")
                for conv, meta in zip(conversations, metadata_list)
            ]

            objective_examples = [
                dspy.Example(patient_convo=conv, patient_metadata=meta).with_inputs(
                    "patient_convo", "patient_metadata")
                for conv, meta in zip(conversations, metadata_list)
            ]

            # Batch extract subjective and objective in parallel
            subjective_task = asyncio.to_thread(
                self.extract_subjective.batch,
                examples=subjective_examples,
                num_threads=min(len(conversations), 10),  # Limit threads
                max_errors=None,
                return_failed_examples=True
            )

            objective_task = asyncio.to_thread(
                self.extract_objective.batch,
                examples=objective_examples,
                num_threads=min(len(conversations), 10),
                max_errors=None,
                return_failed_examples=True
            )

            # Wait for both batches to complete
            (subj_results, subj_failed, subj_errors), (obj_results, obj_failed, obj_errors) = await asyncio.gather(
                subjective_task, objective_task
            )

            # Log any failures in S/O extraction
            if subj_failed:
                logger.warning(
                    f"Failed to extract subjective for {len(subj_failed)} conversations")
            if obj_failed:
                logger.warning(
                    f"Failed to extract objective for {len(obj_failed)} conversations")

            # Create assessment/plan examples from successful extractions
            assessment_examples = []
            # Track (original_index, subj_result, obj_result)
            result_mapping = []

            for i, (subj_res, obj_res) in enumerate(zip(subj_results, obj_results)):
                # Only process if both S and O succeeded
                if subj_res is not None and obj_res is not None:
                    assessment_examples.append(
                        dspy.Example(
                            subjective_section=subj_res.subjective_section,
                            objective_section=obj_res.objective_section,
                            patient_metadata=metadata_list[i]
                        ).with_inputs("subjective_section", "objective_section", "patient_metadata")
                    )
                    result_mapping.append((i, subj_res, obj_res))
                else:
                    logger.warning(
                        f"Skipping assessment/plan for conversation {i} due to S/O extraction failure")

            # Batch generate assessment and plan for successful extractions
            assessment_results = []
            if assessment_examples:
                assessment_batch = await asyncio.to_thread(
                    self.generate_assessment_plan.batch,
                    examples=assessment_examples,
                    num_threads=min(len(assessment_examples), 10),
                    max_errors=None,
                    return_failed_examples=True
                )
                assessment_results, assess_failed, assess_errors = assessment_batch

                if assess_failed:
                    logger.warning(
                        f"Failed to generate assessment/plan for {len(assess_failed)} conversations")

            # Construct final results, maintaining index alignment
            # Pre-allocate with correct length
            final_results = [None] * len(conversations)

            # Fill in successful results at their original indices
            for (idx, subj_res, obj_res), assess_res in zip(result_mapping, assessment_results):
                if assess_res is not None:  # Assessment succeeded
                    final_results[idx] = {
                        'subjective': subj_res.subjective_section,
                        'objective': obj_res.objective_section,
                        'assessment': assess_res.assessment_section,
                        'plan': assess_res.plan_section,
                        'engine_type': self.engine_type,
                        'version': self.version,
                        'original_index': idx
                    }
                else:
                    # Assessment failed
                    final_results[idx] = {
                        'error': 'Assessment/plan generation failed',
                        'original_index': idx
                    }

            # Fill in errors for conversations that failed at S/O extraction
            for i, result in enumerate(final_results):
                if result is None:
                    final_results[i] = {
                        'error': 'Subjective/objective extraction failed',
                        'original_index': i
                    }

            logger.info(
                f"Completed batch SOAP generation: {sum(1 for r in final_results if 'error' not in r)}/{len(conversations)} succeeded")
            return final_results

        except Exception as e:
            logger.error(f"Batch SOAP generation failed: {e}")
            raise SOAPGenerationError(f"Batch SOAP generation failed: {e}")


# ==================== LLM ENGINE ====================

class LLMSOAPEngine(SOAPEngine):
    """
    LLM-based SOAP note generation engine.

    Uses a language model API to generate complete SOAP notes in a single call.
    Supports configurable prompts via YAML file with separate system and user messages.
    """

    def __init__(self, llm_client, model_name: str = "gpt-4", prompt_file: str = "config/llm_prompts.yaml"):
        """
        Initialize LLM engine with client and prompt configuration.

        Args:
            llm_client: LLM client instance (OpenAI or compatible)
            model_name: Name of the model to use
            prompt_file: Path to YAML file with system and user prompts
        """
        self.llm_client = llm_client
        self.model_name = model_name
        self.engine_type = "llm"
        self.version = "1.0"
        self.prompts = self._load_prompts(prompt_file)

    def _load_prompts(self, prompt_file: str) -> Dict[str, str]:
        """
        Load system and user prompts from YAML file.

        Args:
            prompt_file: Path to YAML file with 'system_prompt' and 'user_prompt' keys

        Returns:
            Dictionary with 'system' and 'user' prompt templates

        Raises:
            FileNotFoundError: If prompt file doesn't exist
            ValueError: If required keys are missing
        """
        if not Path(prompt_file).exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

        try:
            with open(prompt_file, 'r') as f:
                config = yaml.safe_load(f)

            # Validate required keys
            if 'system_prompt' not in config or 'user_prompt' not in config:
                raise ValueError(
                    "Prompt file must contain 'system_prompt' and 'user_prompt' keys")

            logger.info(f"Loaded prompts from {prompt_file}")
            return {
                'system': config['system_prompt'],
                'user': config['user_prompt']
            }

        except yaml.YAMLError as e:
            logger.error(f"Failed to parse YAML file {prompt_file}: {e}")
            raise ValueError(f"Invalid YAML format in {prompt_file}: {e}")

    async def _call_llm_async(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call LLM API asynchronously.

        Args:
            system_prompt: System message defining assistant behavior
            user_prompt: User message with the actual request

        Returns:
            Generated text from the LLM

        Raises:
            SOAPGenerationError: If LLM call fails
        """
        try:
            if hasattr(self.llm_client, 'chat'):  # OpenAI style
                response = await asyncio.to_thread(
                    self.llm_client.chat.completions.create,
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.1
                )
                return response.choices[0].message.content
            else:  # Generic client with system/user support
                return await asyncio.to_thread(
                    self.llm_client.generate,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.model_name
                )

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise SOAPGenerationError(f"LLM call failed: {e}")

    def _build_prompts(self, patient_convo: str, patient_metadata: str) -> tuple[str, str]:
        """
        Build system and user prompts from templates.

        Args:
            patient_convo: Patient-provider conversation transcript
            patient_metadata: Patient demographics and background

        Returns:
            Tuple of (system_prompt, user_prompt)
        """
        system_prompt = self.prompts['system']
        user_prompt = self.prompts['user'].format(
            patient_metadata=patient_metadata,
            patient_convo=patient_convo
        )
        return system_prompt, user_prompt

    def _parse_soap_sections(self, soap_note: str) -> Dict[str, str]:
        """
        Parse SOAP note text into individual sections.

        Uses regex to extract SUBJECTIVE, OBJECTIVE, ASSESSMENT, and PLAN sections.
        Falls back to empty strings if sections cannot be parsed.

        Args:
            soap_note: Complete SOAP note as text

        Returns:
            Dictionary with keys: subjective, objective, assessment, plan
        """
        import re
        sections = {
            'subjective': '',
            'objective': '',
            'assessment': '',
            'plan': ''
        }

        # Try to extract sections using regex
        subj_match = re.search(
            r'SUBJECTIVE[:\s]+(.*?)(?=OBJECTIVE|ASSESSMENT|PLAN|$)',
            soap_note,
            re.DOTALL | re.IGNORECASE
        )
        obj_match = re.search(
            r'OBJECTIVE[:\s]+(.*?)(?=ASSESSMENT|PLAN|$)',
            soap_note,
            re.DOTALL | re.IGNORECASE
        )
        assess_match = re.search(
            r'ASSESSMENT[:\s]+(.*?)(?=PLAN|$)',
            soap_note,
            re.DOTALL | re.IGNORECASE
        )
        plan_match = re.search(
            r'PLAN[:\s]+(.*?)$',
            soap_note,
            re.DOTALL | re.IGNORECASE
        )

        if subj_match:
            sections['subjective'] = subj_match.group(1).strip()
        if obj_match:
            sections['objective'] = obj_match.group(1).strip()
        if assess_match:
            sections['assessment'] = assess_match.group(1).strip()
        if plan_match:
            sections['plan'] = plan_match.group(1).strip()

        # Warn if parsing failed
        if not any(sections.values()):
            logger.warning("Failed to parse SOAP sections from LLM output")

        return sections

    async def generate_soap_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """
        Generate single SOAP note using LLM.

        Args:
            patient_convo: Patient-provider conversation transcript
            patient_metadata: Patient demographics and background

        Returns:
            Dictionary with SOAP sections and full_note

        Raises:
            SOAPGenerationError: If generation fails
        """
        try:
            system_prompt, user_prompt = self._build_prompts(
                patient_convo, patient_metadata)
            soap_note = await self._call_llm_async(system_prompt, user_prompt)
            sections = self._parse_soap_sections(soap_note)

            return {
                'subjective': sections['subjective'],
                'objective': sections['objective'],
                'assessment': sections['assessment'],
                'plan': sections['plan'],
                'full_note': soap_note,
                'engine_type': self.engine_type,
                'version': self.version
            }

        except Exception as e:
            logger.error(f"LLM SOAP generation failed: {e}")
            raise SOAPGenerationError(f"LLM SOAP generation failed: {e}")

    async def generate_soap_batch_async(self, conversations: List[str], metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple SOAP notes with concurrent LLM calls.

        Uses semaphore to limit concurrent requests and prevent rate limiting.
        Handles exceptions gracefully to avoid batch failures.

        Args:
            conversations: List of conversation transcripts
            metadata_list: List of patient metadata

        Returns:
            List of SOAP note dictionaries (errors included with 'error' key)

        Raises:
            SOAPGenerationError: If batch setup fails
        """
        try:
            if len(conversations) != len(metadata_list):
                raise ValueError(
                    "Conversations and metadata lists must have same length")

            logger.info(
                f"Starting batch LLM SOAP generation for {len(conversations)} conversations")

            # Create tasks for all SOAP generations
            tasks = [
                self.generate_soap_async(conv, meta)
                for conv, meta in zip(conversations, metadata_list)
            ]

            # Process all in parallel with semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(10)  # Max 10 concurrent LLM calls

            async def limited_task(task):
                """Execute task with semaphore limiting."""
                async with semaphore:
                    return await task

            results = await asyncio.gather(
                *[limited_task(task) for task in tasks],
                return_exceptions=True
            )

            # Handle exceptions and add original indices
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(
                        f"LLM generation failed for conversation {i}: {result}")
                    final_results.append({
                        'error': str(result),
                        'original_index': i
                    })
                else:
                    result['original_index'] = i
                    final_results.append(result)

            successful = sum(1 for r in final_results if 'error' not in r)
            logger.info(
                f"Completed batch LLM generation: {successful}/{len(conversations)} succeeded")

            return final_results

        except Exception as e:
            logger.error(f"Batch LLM SOAP generation failed: {e}")
            raise SOAPGenerationError(f"Batch LLM SOAP generation failed: {e}")


# ==================== MAIN PIPELINE ====================

class SOAPGenerationPipeline:
    """
    Main pipeline for SOAP note generation.

    Supports multiple engines (DSPy, LLM) with easy switching and consistent API.
    """

    def __init__(self, engine_type: str = "dspy", **engine_kwargs):
        """
        Initialize pipeline with specified engine.

        Args:
            engine_type: "dspy" or "llm"
            **engine_kwargs: Additional arguments for engine initialization
                For LLM engine: llm_client, model_name, prompt_file
        """
        if engine_type == "dspy":
            self.engine = DSPySOAPEngine()
        elif engine_type == "llm":
            self.engine = LLMSOAPEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        self.engine_type = engine_type
        logger.info(f"Initialized SOAP pipeline with {engine_type} engine")

    async def forward_async(self, patient_convo: str, patient_metadata: str) -> Dict[str, Any]:
        """
        Generate single SOAP note.

        Args:
            patient_convo: Patient-provider conversation transcript
            patient_metadata: Patient demographics and background

        Returns:
            Dictionary with SOAP sections and pipeline metadata
        """
        result = await self.engine.generate_soap_async(patient_convo, patient_metadata)
        result['pipeline_info'] = {
            'pipeline_version': '2.0',
            'engine_used': self.engine_type
        }
        return result

    async def forward_batch_async(self, conversations: List[str], metadata_list: List[str]) -> List[Dict[str, Any]]:
        """
        Generate multiple SOAP notes in batch.

        Args:
            conversations: List of conversation transcripts
            metadata_list: List of patient metadata

        Returns:
            List of dictionaries with SOAP sections and pipeline metadata
        """
        results = await self.engine.generate_soap_batch_async(conversations, metadata_list)

        # Add pipeline info to all results
        for result in results:
            if 'error' not in result:
                result['pipeline_info'] = {
                    'pipeline_version': '2.0',
                    'engine_used': self.engine_type
                }

        return results

    def switch_engine(self, engine_type: str, **engine_kwargs) -> None:
        """
        Switch to a different generation engine.

        Args:
            engine_type: "dspy" or "llm"
            **engine_kwargs: Additional arguments for engine initialization
        """
        if engine_type == "dspy":
            self.engine = DSPySOAPEngine()
        elif engine_type == "llm":
            self.engine = LLMSOAPEngine(**engine_kwargs)
        else:
            raise ValueError(f"Unknown engine type: {engine_type}")

        self.engine_type = engine_type
        logger.info(f"Switched to {engine_type} engine")
