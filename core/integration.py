"""
Simple Integration Layer for SOAP System
========================================

Connects SOAP generator, evaluator, and storage without overcomplicating.
Works with your existing loader.py and main.py structure.
"""

from typing import Dict, Any, Optional, List
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

    def process_single(self, conversation: str, metadata: str, source_name: str = "unknown") -> Dict[str, Any]:
        """Process one conversation through the pipeline"""

        # Generate SOAP note
        soap_result = self.soap_pipeline.forward(conversation, metadata)

        # Add required fields for storage
        soap_result['original_transcript'] = conversation
        soap_result['patient_metadata'] = metadata
        soap_result['source_name'] = source_name

        # Run evaluation if enabled
        if self.evaluator is not None:
            if self.evaluation_mode == "deterministic":
                eval_result = self.evaluator.evaluate_deterministic(
                    conversation, soap_result['complete_soap_note'])
            elif self.evaluation_mode == "llm_only":
                eval_result = self.evaluator.evaluate_llm_only(
                    conversation, soap_result['complete_soap_note'])
            else:  # comprehensive
                eval_result = self.evaluator.evaluate_comprehensive(
                    conversation, soap_result['complete_soap_note'])

            soap_result['evaluation_metrics'] = eval_result

        # Save to storage
        self.storage.save_result(soap_result)

        return soap_result

    def process_normalized_data(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """Process data that's already been normalized by your loader"""
        results = []

        for item in normalized_data:
            conversation = item.get('transcript', '')
            metadata = str(item.get('patient_metadata', {}))

            if conversation:  # Only process if we have a conversation
                result = self.process_single(
                    conversation, metadata, source_name)
                results.append(result)

        return results

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
