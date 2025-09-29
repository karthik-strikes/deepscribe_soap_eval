"""
Simple Integration Layer for SOAP System with True Async/Batch Processing
========================================

Connects SOAP generator, evaluator, and storage with proper async handling.
"""

import asyncio
import aiofiles
import json
import logging
from typing import Dict, Any, Optional, List
from tqdm.asyncio import tqdm as async_tqdm
from core.soap_generator import SOAPGenerationPipeline
from evaluation.evaluator import EvaluationPipeline
from core.storage import FlexibleSOAPStorage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleSOAPIntegration:
    """
    Integration layer with true async/batch processing capabilities.

    Orchestrates SOAP note generation, evaluation, and storage with optimized
    batch processing and duplicate detection.
    """

    def __init__(self,
                 soap_engine: str = "dspy",
                 evaluation_mode: str = "comprehensive",
                 storage_mode: str = "both",
                 storage_file: str = "results/soap_results.json",
                 batch_size: int = 10,
                 **engine_kwargs):
        """
        Initialize integration with async-ready components.

        Args:
            soap_engine: Type of SOAP generation engine ("dspy" or "llm")
            evaluation_mode: Evaluation strategy ("skip", "deterministic", "llm_only", "comprehensive")
            storage_mode: Storage strategy ("both", "soap_only", etc.)
            storage_file: Path to results file
            batch_size: Number of items to process per batch
            **engine_kwargs: Additional arguments for the SOAP engine
        """
        self.soap_pipeline = SOAPGenerationPipeline(
            engine_type=soap_engine, **engine_kwargs)

        # Configure evaluator based on mode
        if evaluation_mode == "skip":
            self.evaluator = None
        elif evaluation_mode == "deterministic":
            self.evaluator = EvaluationPipeline(llm_evaluators=[])
        elif evaluation_mode == "llm_only":
            self.evaluator = EvaluationPipeline(deterministic_evaluators=[])
        else:  # comprehensive
            self.evaluator = EvaluationPipeline()

        # Wrap storage with async adapter
        base_storage = FlexibleSOAPStorage(
            storage_file=storage_file, mode=storage_mode)
        self.storage = AsyncStorageWrapper(base_storage)

        self.evaluation_mode = evaluation_mode
        self.batch_size = batch_size

        # Cache for duplicate detection
        self._duplicate_cache = set()
        self._load_duplicate_cache()

    def _load_duplicate_cache(self) -> None:
        """
        Load existing records into cache for fast duplicate detection.

        Populates in-memory cache with hashes of previously processed records
        to avoid reprocessing duplicates without hitting storage.
        """
        try:
            existing_records = self.storage.load_all_results()
            for record in existing_records:
                # Handle both 'original_transcript' and 'conversation' keys
                transcript = record.get(
                    'original_transcript') or record.get('conversation', '')
                cache_key = self._make_cache_key(
                    transcript,
                    record.get('patient_metadata', '')
                )
                self._duplicate_cache.add(cache_key)
            logger.info(
                f"Loaded {len(self._duplicate_cache)} existing records into cache")
        except FileNotFoundError:
            logger.info("No existing records found, starting with empty cache")
        except json.JSONDecodeError as e:
            logger.error(f"Error reading existing records: {e}")
        except Exception as e:
            logger.error(f"Unexpected error loading duplicate cache: {e}")

    def _make_cache_key(self, conversation: str, metadata: str) -> str:
        """
        Create deterministic cache key for duplicate detection.

        Args:
            conversation: The conversation transcript
            metadata: Patient metadata (as string)

        Returns:
            MD5 hash of the combined conversation and metadata
        """
        import hashlib
        # Ensure consistent serialization
        combined = f"{conversation}|{metadata}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _is_duplicate_fast(self, conversation: str, metadata: str) -> bool:
        """
        Fast in-memory duplicate check using cache.

        Args:
            conversation: The conversation transcript
            metadata: Patient metadata (as string)

        Returns:
            True if this conversation has already been processed
        """
        cache_key = self._make_cache_key(conversation, metadata)
        return cache_key in self._duplicate_cache

    def _mark_as_processed(self, conversation: str, metadata: str) -> None:
        """
        Mark item as processed in cache.

        Args:
            conversation: The conversation transcript
            metadata: Patient metadata (as string)
        """
        cache_key = self._make_cache_key(conversation, metadata)
        self._duplicate_cache.add(cache_key)

    def _build_soap_note_from_sections(self, soap_result: Dict[str, Any]) -> str:
        """
        Build complete SOAP note from individual sections.

        Args:
            soap_result: Dictionary containing SOAP sections

        Returns:
            Formatted SOAP note as a single string
        """
        sections = []

        if 'subjective' in soap_result and soap_result['subjective']:
            sections.append(f"SUBJECTIVE:\n{soap_result['subjective']}")

        if 'objective' in soap_result and soap_result['objective']:
            sections.append(f"OBJECTIVE:\n{soap_result['objective']}")

        if 'assessment' in soap_result and soap_result['assessment']:
            sections.append(f"ASSESSMENT:\n{soap_result['assessment']}")

        if 'plan' in soap_result and soap_result['plan']:
            sections.append(f"PLAN:\n{soap_result['plan']}")

        if not sections:
            # Fallback to full_note if available (LLM engine)
            return soap_result.get('full_note', "Generated SOAP note (sections not available)")

        return "\n\n".join(sections)

    async def process_single_async(self, conversation: str, metadata: str, source_name: str = "unknown") -> Optional[Dict[str, Any]]:
        """
        Process a single conversation asynchronously.

        Args:
            conversation: The conversation transcript
            metadata: Patient metadata (as string)
            source_name: Source identifier for tracking

        Returns:
            Result dictionary with SOAP note and evaluation, or None if duplicate
        """
        # Fast duplicate check before expensive operations
        if self._is_duplicate_fast(conversation, metadata):
            logger.debug(f"Skipping duplicate from {source_name}")
            return None

        try:
            # Generate SOAP note
            soap_result = await self.soap_pipeline.forward_async(conversation, metadata)

            # Add required tracking fields
            soap_result['conversation'] = conversation
            soap_result['source_name'] = source_name

            # Run evaluation if enabled
            if self.evaluator is not None:
                generated_note = self._build_soap_note_from_sections(
                    soap_result)
                eval_result = await self.evaluator.evaluate_async(
                    conversation, generated_note, metadata, self.evaluation_mode
                )
                soap_result['evaluation_metrics'] = eval_result

            # Save result and mark as processed atomically
            try:
                await self.storage.save_result_async(soap_result)
                self._mark_as_processed(conversation, metadata)
            except Exception as save_error:
                logger.error(f"Failed to save result: {save_error}")
                # Don't mark as processed if save failed
                raise

            return soap_result

        except Exception as e:
            logger.error(
                f"Error processing conversation from {source_name}: {e}")
            return {
                'error': str(e),
                'conversation': conversation,
                'source_name': source_name
            }

    async def process_batch_async(self, items: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """
        Process batch using true batch operations (not just parallel singles).

        Output format:
        {
            "conversation": "Doctor: How are you...",
            "generated_soap": "SUBJECTIVE: Patient reports...",
            "compared_on": "ground_truth" or "transcript",
            "ground_truth": "..." (only if GT exists),
            "evaluation_metrics": {...},
            "source_name": "...",
            "patient_metadata": {...}
        }

        Args:
            items: List of items with 'transcript' and 'patient_metadata' fields
            source_name: Source identifier for tracking

        Returns:
            List of result dictionaries
        """
        # Filter out duplicates upfront to avoid wasted processing
        non_duplicate_items = []
        for item in items:
            conv = item.get('transcript', '')
            meta = str(item.get('patient_metadata', {}))
            if conv and not self._is_duplicate_fast(conv, meta):
                non_duplicate_items.append(item)

        if not non_duplicate_items:
            logger.debug(
                f"All items in batch from {source_name} are duplicates")
            return []

        # Extract conversations and metadata for batch processing
        conversations = [item.get('transcript', '')
                         for item in non_duplicate_items]
        metadata_list = [str(item.get('patient_metadata', {}))
                         for item in non_duplicate_items]

        try:
            # TRUE BATCH: Generate all SOAP notes in one batch call
            soap_results = await self.soap_pipeline.forward_batch_async(conversations, metadata_list)

            # Build generated SOAP notes as strings
            generated_soaps = []
            for soap_result in soap_results:
                if 'error' not in soap_result:
                    generated_soaps.append(
                        self._build_soap_note_from_sections(soap_result))
                else:
                    generated_soaps.append("")

            # Prepare final results with clear structure
            final_results = []
            for i, (item, soap_result) in enumerate(zip(non_duplicate_items, soap_results)):
                if 'error' in soap_result:
                    final_results.append({
                        'error': soap_result['error'],
                        'conversation': conversations[i],
                        'source_name': source_name
                    })
                    continue

                # Base result structure
                result = {
                    'conversation': conversations[i],
                    'generated_soap': generated_soaps[i],
                    'source_name': source_name,
                    'patient_metadata': item.get('patient_metadata', {})
                }

                # Add SOAP sections for reference
                result.update(soap_result)

                # Add ground truth and comparison tracking
                ground_truth = item.get('ground_truth', '')
                if ground_truth:
                    result['ground_truth'] = ground_truth
                    result['compared_on'] = 'ground_truth'
                else:
                    result['compared_on'] = 'transcript'

                final_results.append(result)

            # Run evaluation in batch if enabled
            if self.evaluator is not None:
                # Determine what to compare against
                eval_conversations = []
                valid_indices = []

                for i, (item, result) in enumerate(zip(non_duplicate_items, final_results)):
                    if 'error' not in result:
                        ground_truth = item.get('ground_truth', '')
                        transcript = item.get('transcript', '')
                        # Use ground_truth if exists, else transcript
                        eval_conversations.append(
                            ground_truth if ground_truth else transcript)
                        valid_indices.append(i)

                if eval_conversations:
                    # Get generated notes and metadata for valid items
                    valid_generated_notes = [generated_soaps[i]
                                             for i in valid_indices]
                    valid_metadata = [metadata_list[i]
                                      for i in valid_indices]

                    # TRUE BATCH: Evaluate all at once
                    eval_results = await self.evaluator.evaluate_batch_async(
                        eval_conversations, valid_generated_notes, valid_metadata, self.evaluation_mode
                    )

                    # Add evaluation results to final results
                    for idx, eval_result in zip(valid_indices, eval_results):
                        final_results[idx]['evaluation_metrics'] = eval_result

            # Save all results asynchronously
            try:
                await self.storage.save_batch_async(final_results)

                # Only mark as processed after successful save
                for i, result in enumerate(final_results):
                    if 'error' not in result:
                        self._mark_as_processed(
                            conversations[i], metadata_list[i])
            except Exception as save_error:
                logger.error(f"Failed to save batch results: {save_error}")
                raise

            return final_results

        except Exception as e:
            logger.error(f"Error processing batch from {source_name}: {e}")
            # Return error for all items in batch
            return [{
                'error': str(e),
                'conversation': conv,
                'source_name': source_name
            } for conv in conversations]

    async def process_normalized_data_async(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """
        Process normalized data in batches with progress tracking.

        Splits data into batches and processes each batch with a progress bar
        to provide user feedback during long-running operations.

        Args:
            normalized_data: List of normalized data items
            source_name: Source identifier for tracking

        Returns:
            List of successfully processed results (errors and duplicates filtered)
        """
        all_results = []
        total_items = len(normalized_data)
        num_batches = (total_items + self.batch_size - 1) // self.batch_size

        logger.info(
            f"Processing {total_items} samples in {num_batches} batches of size {self.batch_size}")

        # Process in batches with single progress bar
        with async_tqdm(total=total_items, desc="Processing progress", unit="sample") as pbar:
            for i in range(0, total_items, self.batch_size):
                batch = normalized_data[i:i + self.batch_size]

                # Process entire batch at once
                batch_results = await self.process_batch_async(batch, source_name)
                all_results.extend(batch_results)

                # Update progress bar after batch completes
                pbar.update(len(batch))

        # Filter out None and error results
        successful_results = [
            r for r in all_results if r is not None and 'error' not in r]
        logger.info(
            f"Successfully processed {len(successful_results)} out of {total_items} items")

        return successful_results

    async def process_evaluation_only_async(self, normalized_data: List[Dict[str, Any]], source_name: str) -> List[Dict[str, Any]]:
        """
        Evaluate existing SOAP notes in batches with intelligent ground truth handling.

        Output format:
        {
            "conversation": "Doctor: How are you...",
            "referenced_soap": "SUBJECTIVE: ...",
            "compared_on": "ground_truth" or "transcript",
            "ground_truth": "..." (only if GT exists),
            "evaluation_metrics": {...},
            "source_name": "...",
            "patient_metadata": {...}
        }

        Logic:
        - If ground_truth exists AND different from referenced_soap: use ground_truth, compared_on="ground_truth"
        - If ground_truth == referenced_soap: use transcript, compared_on="transcript"
        - If only referenced_soap: use transcript, compared_on="transcript"

        Args:
            normalized_data: List of items with 'transcript' and 'reference_notes'
            source_name: Source identifier for tracking

        Returns:
            List of evaluation results
        """
        if self.evaluator is None:
            logger.error("Evaluation requested but no evaluator configured")
            return [{"error": "No evaluator configured"}]

        all_results = []

        # Filter items with required data
        valid_items = [
            item for item in normalized_data
            if item.get('transcript') and item.get('reference_notes')
        ]

        if not valid_items:
            logger.warning(
                "No valid items found for evaluation (need 'transcript' and 'reference_notes')")
            return []

        # Calculate total batches
        num_batches = (len(valid_items) + self.batch_size -
                       1) // self.batch_size
        logger.info(
            f"Evaluating {len(valid_items)} samples in {num_batches} batches")

        # Process in batches with single progress bar
        with async_tqdm(total=len(valid_items), desc="Evaluation progress", unit="sample") as pbar:
            for i in range(0, len(valid_items), self.batch_size):
                batch = valid_items[i:i + self.batch_size]

                # Extract data with intelligent ground truth handling
                eval_sources = []  # What to extract findings from
                referenced_soaps = []  # The SOAP notes being evaluated
                metadata_list = []
                compared_on_list = []  # Track what we compared against

                for item in batch:
                    ref_soap = item.get('reference_notes', '')
                    ground_truth = item.get('ground_truth', '')
                    transcript = item.get('transcript', '')

                    # Decide what to compare against
                    if ground_truth:
                        if ref_soap != ground_truth:
                            # Case 1: GT exists and different from ref_soap
                            eval_sources.append(ground_truth)
                            compared_on_list.append('ground_truth')
                        else:
                            # Case 2: GT exists and same as ref_soap
                            # Use transcript to cross-check
                            eval_sources.append(transcript)
                            compared_on_list.append('transcript')
                    else:
                        # Case 3: No ground truth
                        eval_sources.append(transcript)
                        compared_on_list.append('transcript')

                    referenced_soaps.append(ref_soap)
                    metadata_list.append(str(item.get('patient_metadata', {})))

                try:
                    # TRUE BATCH: Evaluate all at once
                    eval_results = await self.evaluator.evaluate_batch_async(
                        eval_sources, referenced_soaps, metadata_list, self.evaluation_mode
                    )

                    # Construct results with clear structure
                    batch_results = []
                    for j, (item, eval_result) in enumerate(zip(batch, eval_results)):
                        result = {
                            'conversation': item.get('transcript', ''),
                            'referenced_soap': referenced_soaps[j],
                            'compared_on': compared_on_list[j],
                            'evaluation_metrics': eval_result,
                            'source_name': source_name,
                            'patient_metadata': item.get('patient_metadata', {})
                        }

                        # Add ground_truth only if it exists
                        ground_truth = item.get('ground_truth', '')
                        if ground_truth:
                            result['ground_truth'] = ground_truth

                        batch_results.append(result)

                    # Save batch
                    await self.storage.save_batch_async(batch_results)
                    all_results.extend(batch_results)

                except Exception as e:
                    logger.error(f"Error evaluating batch: {e}")
                    # Continue with next batch

                # Update progress bar after batch completes
                pbar.update(len(batch))

        logger.info(f"Completed evaluation of {len(all_results)} items")
        return all_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.

        Returns:
            Dictionary with statistics about stored results
        """
        return self.storage.get_stats()


# ==================== ASYNC STORAGE ADAPTER ====================

class AsyncStorageWrapper:
    """
    Wrapper to add async methods to FlexibleSOAPStorage.

    Provides async save operations with proper locking to ensure thread-safe
    writes to the underlying synchronous storage.
    """

    def __init__(self, storage: FlexibleSOAPStorage):
        """
        Initialize wrapper around synchronous storage.

        Args:
            storage: The FlexibleSOAPStorage instance to wrap
        """
        self.storage = storage
        self._write_lock = asyncio.Lock()

    async def save_result_async(self, result: Dict[str, Any]) -> None:
        """
        Asynchronously save a single result with locking.

        Args:
            result: Result dictionary to save
        """
        async with self._write_lock:
            await asyncio.to_thread(self.storage.save_result, result)

    async def save_batch_async(self, results: List[Dict[str, Any]]) -> None:
        """
        Asynchronously save batch results with locking.

        Args:
            results: List of result dictionaries to save
        """
        async with self._write_lock:
            await asyncio.to_thread(self._save_batch_internal, results)

    def _save_batch_internal(self, results: List[Dict[str, Any]]) -> None:
        """
        Internal batch save implementation.

        Args:
            results: List of result dictionaries to save
        """
        for result in results:
            self.storage.save_result(result)

    def is_duplicate(self, conversation: str, metadata: str) -> bool:
        """
        Check if conversation is a duplicate (passthrough to storage).

        Args:
            conversation: The conversation transcript
            metadata: Patient metadata (as string)

        Returns:
            True if duplicate exists in storage
        """
        return self.storage.is_duplicate(conversation, metadata)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics (passthrough to storage).

        Returns:
            Dictionary with statistics about stored results
        """
        return self.storage.get_stats()

    def load_all_results(self) -> List[Dict[str, Any]]:
        """
        Load all stored results (passthrough to storage).

        Returns:
            List of all stored result dictionaries
        """
        return self.storage.load_all_results()


# ==================== FACTORY FUNCTIONS ====================

def create_fast_integration(storage_file: str = "results/fast_results.json",
                            batch_size: int = 20) -> SimpleSOAPIntegration:
    """
    Create integration optimized for speed.

    Uses DSPy engine with deterministic evaluation only for fast processing
    with good quality.

    Args:
        storage_file: Path to results file
        batch_size: Number of items per batch

    Returns:
        Configured SimpleSOAPIntegration instance
    """
    return SimpleSOAPIntegration(
        soap_engine="dspy",
        evaluation_mode="deterministic",
        storage_mode="both",
        storage_file=storage_file,
        batch_size=batch_size
    )


def create_thorough_integration(llm_client, model_name: str = "gpt-4",
                                prompt_file: str = "config/llm_prompts.yaml",
                                storage_file: str = "results/thorough_results.json",
                                batch_size: int = 10) -> SimpleSOAPIntegration:
    """
    Create integration optimized for quality.

    Uses LLM engine with comprehensive evaluation for highest quality results.

    Args:
        llm_client: LLM client instance
        model_name: Name of the model to use
        prompt_file: Path to YAML file with system/user prompts
        storage_file: Path to results file
        batch_size: Number of items per batch

    Returns:
        Configured SimpleSOAPIntegration instance
    """
    return SimpleSOAPIntegration(
        soap_engine="llm",
        evaluation_mode="comprehensive",
        storage_mode="both",
        storage_file=storage_file,
        llm_client=llm_client,
        model_name=model_name,
        prompt_file=prompt_file,
        batch_size=batch_size
    )


def create_generation_only(engine_type: str = "dspy",
                           storage_file: str = "results/soap_only.json",
                           batch_size: int = 20) -> SimpleSOAPIntegration:
    """
    Create integration for generation without evaluation.

    Optimized for maximum throughput when evaluation is not needed.

    Args:
        engine_type: Type of generation engine ("dspy" or "llm")
        storage_file: Path to results file
        batch_size: Number of items per batch

    Returns:
        Configured SimpleSOAPIntegration instance
    """
    return SimpleSOAPIntegration(
        soap_engine=engine_type,
        evaluation_mode="skip",
        storage_mode="soap_only",
        storage_file=storage_file,
        batch_size=batch_size
    )
