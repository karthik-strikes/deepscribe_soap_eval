#!/usr/bin/env python3
"""DeepScribe SOAP Evaluation System"""

from data.loader import UniversalDataLoader, DSPyFieldDetector
from core.integration import SimpleSOAPIntegration
import argparse
import sys
import dspy
import json
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent))

VALID_OPTIONS = {
    "mode": ["generate", "evaluate", "both"],
    "storage": ["soap_only", "evaluation_only", "both"],
    "soap_engine": ["dspy", "llm"],
    "evaluation_mode": ["deterministic", "llm_only", "comprehensive", "skip"]
}

SAMPLE_CONFIG = {
    "model": {"name": "gemini/gemini-2.5-pro", "max_tokens": 4000, "temperature": 0.1},
    "defaults": {
        "samples": 5,
        "mode": "both",
        "output": "results/soap_results.json",
        "storage": "both",
        "soap_engine": "dspy",
        "evaluation_mode": "comprehensive"
    }
}


def load_config(config_file="config.json"):
    if not os.path.exists(config_file):
        return None
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except:
        return None


def create_sample_config(filename="config.json"):
    if os.path.exists(filename):
        print(f"Config {filename} already exists")
        return False
    with open(filename, 'w') as f:
        json.dump(SAMPLE_CONFIG, f, indent=2)
    print(f"Created config: {filename}")
    return True


def setup_dspy_model(model_name, max_tokens=4000, temperature=0.1):
    try:
        lm = dspy.LM(model_name, max_tokens=max_tokens,
                     temperature=temperature)
        dspy.configure(lm=lm)
        return True
    except Exception as e:
        print(f"Model setup failed: {e}")
        return False


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DeepScribe SOAP Evaluation System")
    parser.add_argument("--source", help="Data source")
    parser.add_argument("--samples", type=int, help="Number of samples")
    parser.add_argument(
        "--mode", choices=VALID_OPTIONS["mode"], help="Processing mode")
    parser.add_argument("--output", help="Output file")
    parser.add_argument("--model", help="Model to use")
    parser.add_argument(
        "--storage", choices=VALID_OPTIONS["storage"], help="Storage mode")
    parser.add_argument(
        "--soap-engine", choices=VALID_OPTIONS["soap_engine"], help="SOAP generation engine")
    parser.add_argument(
        "--evaluation-mode", choices=VALID_OPTIONS["evaluation_mode"], help="Evaluation mode")
    parser.add_argument("--config", default="config.json", help="Config file")
    parser.add_argument("--create-config", action="store_true",
                        help="Create sample config")

    args = parser.parse_args()

    if args.create_config:
        return 0 if create_sample_config(args.config) else 1

    # Load config
    config = load_config(args.config)
    model_config = config.get("model", {}) if config else {}
    defaults_config = config.get("defaults", {}) if config else {}

    # Get final values (CLI > config > required error)
    source = args.source
    model_name = args.model or model_config.get("name")
    samples = args.samples or defaults_config.get("samples")
    mode = args.mode or defaults_config.get("mode")
    storage_mode = args.storage or defaults_config.get("storage")
    soap_engine = args.soap_engine or defaults_config.get("soap_engine")
    evaluation_mode = args.evaluation_mode or defaults_config.get(
        "evaluation_mode")
    output = args.output or defaults_config.get("output")

    # Check required parameters
    if not all([source, model_name, samples, mode, storage_mode, soap_engine, evaluation_mode, output]):
        print("Missing required parameters. Use --create-config or provide all via CLI")
        return 1

    # Create output directory if needed
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)

    print(f"Source: {source}, Model: {model_name}, Samples: {samples}")
    print(
        f"Mode: {mode}, SOAP Engine: {soap_engine}, Evaluation: {evaluation_mode}, Storage: {storage_mode}")

    # Setup model
    if not setup_dspy_model(model_name, model_config.get("max_tokens", 4000), model_config.get("temperature", 0.1)):
        return 1

    # Initialize components
    detector = DSPyFieldDetector()
    loader = UniversalDataLoader(detector)

    try:
        # Load and normalize data
        normalized_data, field_mapping = loader.load_and_normalize(
            source=source, max_samples=samples)
        print(f"Loaded {len(normalized_data)} samples")
        print(
            f"Field mapping confidence: {field_mapping.confidence_score:.2f}")

        # Initialize integration pipeline
        integration = SimpleSOAPIntegration(
            soap_engine=soap_engine,
            evaluation_mode=evaluation_mode,
            storage_mode=storage_mode,
            storage_file=output
        )

        # Process based on mode
        results = []

        if mode == "generate":
            print("Generating SOAP notes only...")
            for item in normalized_data:
                conversation = item.get('transcript', '')
                metadata = str(item.get('patient_metadata', {}))

                if conversation:
                    result = integration.process_single(
                        conversation, metadata, source)
                    results.append(result)

        elif mode == "evaluate":
            print("Evaluating reference notes only...")
            for item in normalized_data:
                conversation = item.get('transcript', '')
                reference_note = item.get('reference_notes', '')

                if conversation and reference_note:
                    eval_result = integration.evaluate_existing_note(
                        conversation, reference_note)
                    result = {
                        'original_transcript': conversation,
                        'reference_notes': reference_note,
                        'evaluation_metrics': eval_result,
                        'source_name': source,
                        'patient_metadata': item.get('patient_metadata', {})
                    }
                    # Save to storage
                    integration.storage.save_result(result)
                    results.append(result)

        else:  # both
            print("Generating and evaluating SOAP notes...")
            results = integration.process_normalized_data(
                normalized_data, source)

        # Print summary
        stats = integration.get_stats()
        print(f"\nProcessing complete:")
        print(f"- Processed: {len(results)} samples")
        print(f"- Total stored: {stats.get('total_results', 0)}")
        print(f"- Results saved to: {output}")

        if stats.get('detected_evaluation_fields'):
            print(
                f"- Evaluation metrics: {len(stats['detected_evaluation_fields'])}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
