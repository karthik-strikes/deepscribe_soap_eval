"""
Simple Integration Layer for SOAP System
========================================

Connects SOAP generator, evaluator, and storage without overcomplicating.
Works with your existing loader.py and main.py structure.
"""

import asyncio
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm
from core.soap_generator import SOAPGenerationPipeline
from evaluation.evaluator import EvaluationPipeline
from core.storage import FlexibleSOAPStorage


class SimpleSOAPIntegration:
    """Simple integration that connects your existing components"""

    def __init__(self,
                 soap_engine: str = "dspy",
                 evaluation_mode: str = "comprehensive",
                 storage_mode: str = "both",
                 storage_file: str = "results/soap_results.json",
                 **engine_kwargs):
        """
        Initialize with your existing components.

        Args:
            soap_engine: "dspy" or "llm"
            evaluation_mode: "deterministic", "llm_only", "comprehensive", or "skip"
            storage_mode: "soap_only", "evaluation_only", or "both"
            storage_file: Where to save results
            **engine_kwargs: For LLM engine (llm_client, model_name)
        """
        # Initialize SOAP generator
        self.soap_pipeline = SOAPGenerationPipeline(
            engine_type=soap_engine, **engine_kwargs)

        # Initialize evaluator
        if evaluation_mode == "skip":
            self.evaluator = None
        elif evaluation_mode == "deterministic":
            self.evaluator = EvaluationPipeline(llm_evaluators=[])
        elif evaluation_mode == "llm_only":
            self.evaluator = EvaluationPipeline(deterministic_evaluators=[])
        else:  # comprehensive
            self.evaluator = EvaluationPipeline()

        # Initialize storage
        self.storage = FlexibleSOAPStorage(
            storage_file=storage_file, mode=storage_mode)

        self.evaluation_mode = evaluation_mode

    def _build_soap_note_from_sections(self, soap_result: Dict[str, Any]) -> str:
        """Build a complete SOAP note from individual sections for evaluation"""
        sections = []

        if 'subjective' in soap_result and soap_result['subjective']:
            sections.append(f"SUBJECTIVE:\n{soap_result['subjective']}")

        if 'objective' in soap_result and soap_result['objective']:
            sections.append(f"OBJECTIVE:\n{soap_result['objective']}")

        if 'assessment' in soap_result and soap_result['assessment']:
            sections.append(f"ASSESSMENT:\n{soap_result['assessment']}")

        if 'plan' in soap_result and soap_result['plan']:
            sections.append(f"PLAN:\n{soap_result['plan']}")

        # If no sections available, return a basic note
        if not sections:
            return "Generated SOAP note (sections not available)"

        return "\n\n".join(sections)

    def process_single(self, conversation: str, metadata: str, source_name: str = "unknown") -> Dict[str, Any]:
        """Process one conversation through the pipeline"""

        # Check for duplicates BEFORE expensive processing
        if self.storage.is_duplicate(conversation, metadata):
            return None  # Skip processing duplicates

        # Generate SOAP note
        soap_result = self.soap_pipeline.forward(conversation, metadata)

        # Add required fields for storage
        soap_result['original_transcript'] = conversation
        soap_result['patient_metadata'] = metadata
        soap_result['source_name'] = source_name

        # Run evaluation if enabled
        if self.evaluator is not None:
            # Build generated note from SOAP sections for evaluation
            generated_note = self._build_soap_note_from_sections(soap_result)

            if self.evaluation_mode == "deterministic":
                eval_result = self.evaluator.evaluate_deterministic(
                    conversation, generated_note, metadata)
            elif self.evaluation_mode == "llm_only":
                eval_result = self.evaluator.evaluate_llm_only(
                    conversation, generated_note, metadata)
            else:  # comprehensive
                eval_result = self.evaluator.evaluate_comprehensive(
                    conversation, generated_note, metadata)

            soap_result['evaluation_metrics'] = eval_result

        # Save to storage
        self.storage.save_result(soap_result)

        return soap_result

    def process_normalized_data(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """Sync wrapper - calls async version for optimal performance"""
        return asyncio.run(self.process_normalized_data_async(normalized_data, source_name))

    async def process_single_async(self, conversation: str, metadata: str, source_name: str = "unknown") -> Dict[str, Any]:
        """Async version of process_single for better performance"""

        # Check for duplicates BEFORE expensive processing
        if self.storage.is_duplicate(conversation, metadata):
            return None  # Skip processing duplicates

        # Generate SOAP note using async for better performance
        soap_result = await self.soap_pipeline.forward_async(conversation, metadata)

        # Add required fields for storage
        soap_result['original_transcript'] = conversation
        soap_result['patient_metadata'] = metadata
        soap_result['source_name'] = source_name

        # Run evaluation if enabled (async for better performance)
        if self.evaluator is not None:
            # Build generated note from SOAP sections for evaluation
            generated_note = self._build_soap_note_from_sections(soap_result)

            if self.evaluation_mode == "deterministic":
                eval_result = self.evaluator.evaluate_deterministic(
                    conversation, generated_note, metadata)
            elif self.evaluation_mode == "llm_only":
                if hasattr(self.evaluator, 'evaluate_llm_only_async'):
                    eval_result = await self.evaluator.evaluate_llm_only_async(
                        conversation, generated_note, metadata)
                else:
                    eval_result = self.evaluator.evaluate_llm_only(
                        conversation, generated_note, metadata)
            else:  # comprehensive
                if hasattr(self.evaluator, 'evaluate_comprehensive_async'):
                    eval_result = await self.evaluator.evaluate_comprehensive_async(
                        conversation, generated_note, metadata)
                else:
                    eval_result = self.evaluator.evaluate_comprehensive(
                        conversation, generated_note, metadata)

            soap_result['evaluation_metrics'] = eval_result

        # Save to storage
        self.storage.save_result(soap_result)

        return soap_result

    async def process_normalized_data_async(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """Async version of process_normalized_data for parallel processing"""

        async def process_with_progress(item, pbar):
            conversation = item.get('transcript', '')
            metadata = str(item.get('patient_metadata', {}))

            result = None
            if conversation:
                result = await self.process_single_async(conversation, metadata, source_name)

            pbar.update(1)
            return result

        # Create progress bar
        pbar = tqdm(total=len(normalized_data),
                    desc=f"Processing {len(normalized_data)} samples")

        # Process all items in parallel
        tasks = [process_with_progress(item, pbar) for item in normalized_data]
        results = await asyncio.gather(*tasks)

        pbar.close()

        # Filter out None results
        return [r for r in results if r is not None]

    def process_evaluation_only(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """Sync wrapper for evaluate-only mode with progress bar"""
        return asyncio.run(self.process_evaluation_only_async(normalized_data, source_name))

    async def process_evaluation_only_async(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """Async evaluation-only processing with progress bar"""

        async def evaluate_with_progress(item, pbar):
            conversation = item.get('transcript', '')
            reference_note = item.get('reference_notes', '')
            patient_metadata = item.get('patient_metadata', {})

            result = None
            if conversation and reference_note:
                result = await self.evaluate_existing_note_async(
                    conversation, reference_note, patient_metadata, source_name, item)

            pbar.update(1)
            return result

        # Create progress bar
        pbar = tqdm(total=len(normalized_data),
                    desc=f"Evaluating {len(normalized_data)} samples")

        # Process all items in parallel
        tasks = [evaluate_with_progress(item, pbar)
                 for item in normalized_data]
        results = await asyncio.gather(*tasks)

        pbar.close()

        # Filter out None results
        return [r for r in results if r is not None]

    async def evaluate_existing_note_async(self, conversation: str, existing_soap_note: str,
                                           patient_metadata: Dict[str, Any], source_name: str,
                                           original_item: Dict[str, Any]) -> Dict[str, Any]:
        """Async version of evaluate_existing_note"""
        if self.evaluator is None:
            return {"error": "No evaluator configured"}

        metadata_str = str(patient_metadata)

        if self.evaluation_mode == "deterministic":
            eval_result = self.evaluator.evaluate_deterministic(
                conversation, existing_soap_note, metadata_str)
        elif self.evaluation_mode == "llm_only":
            if hasattr(self.evaluator, 'evaluate_llm_only_async'):
                eval_result = await self.evaluator.evaluate_llm_only_async(conversation, existing_soap_note, metadata_str)
            else:
                eval_result = self.evaluator.evaluate_llm_only(
                    conversation, existing_soap_note, metadata_str)
        else:  # comprehensive
            if hasattr(self.evaluator, 'evaluate_comprehensive_async'):
                eval_result = await self.evaluator.evaluate_comprehensive_async(conversation, existing_soap_note, metadata_str)
            else:
                eval_result = self.evaluator.evaluate_comprehensive(
                    conversation, existing_soap_note, metadata_str)

        result = {
            'original_transcript': conversation,
            'reference_notes': existing_soap_note,
            'evaluation_metrics': eval_result,
            'source_name': source_name,
            'patient_metadata': patient_metadata
        }

        # Save to storage
        self.storage.save_result(result)
        return result

    def evaluate_existing_note(self, conversation: str, existing_soap_note: str) -> Dict[str, Any]:
        """Evaluate an existing SOAP note without regenerating"""
        if self.evaluator is None:
            return {"error": "No evaluator configured"}

        if self.evaluation_mode == "deterministic":
            return self.evaluator.evaluate_deterministic(conversation, existing_soap_note)
        elif self.evaluation_mode == "llm_only":
            return self.evaluator.evaluate_llm_only(conversation, existing_soap_note)
        else:  # comprehensive
            return self.evaluator.evaluate_comprehensive(conversation, existing_soap_note)

    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        return self.storage.get_stats()


# Simple factory functions for common setups
def create_fast_integration(storage_file: str = "results/fast_results.json") -> SimpleSOAPIntegration:
    """Fast processing: DSPy + deterministic evaluation"""
    return SimpleSOAPIntegration(
        soap_engine="dspy",
        evaluation_mode="deterministic",
        storage_mode="both",
        storage_file=storage_file
    )


def create_thorough_integration(llm_client, model_name: str = "gpt-4",
                                storage_file: str = "results/thorough_results.json") -> SimpleSOAPIntegration:
    """Thorough processing: LLM + comprehensive evaluation"""
    return SimpleSOAPIntegration(
        soap_engine="llm",
        evaluation_mode="comprehensive",
        storage_mode="both",
        storage_file=storage_file,
        llm_client=llm_client,
        model_name=model_name
    )


def create_generation_only(engine_type: str = "dspy",
                           storage_file: str = "results/soap_only.json") -> SimpleSOAPIntegration:
    """Generation only: No evaluation"""
    return SimpleSOAPIntegration(
        soap_engine=engine_type,
        evaluation_mode="skip",
        storage_mode="soap_only",
        storage_file=storage_file
    )
