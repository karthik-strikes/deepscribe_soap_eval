#!/usr/bin/env python3
"""
DeepScribe SOAP Evaluation System with Async Support
====================================================

Command-line interface for SOAP note generation and evaluation.
Supports multiple engines (DSPy, LLM) and evaluation modes.
"""

from data.loader import UniversalDataLoader, DSPyFieldDetector
from core.integration import SimpleSOAPIntegration
from utils.model_setup import (
    setup_dspy_model,
    create_llm_client,
    validate_model_config,
    get_default_config
)
import argparse
import sys
import json
import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))

# Valid option values for CLI validation
VALID_OPTIONS = {
    "mode": ["generate", "evaluate", "both"],
    "storage": ["soap_only", "evaluation_only", "both"],
    "soap_engine": ["dspy", "llm"],
    "evaluation_mode": ["deterministic", "llm_only", "comprehensive", "skip"]
}


def load_config(config_file: str = "config.json") -> dict:
    """Load configuration from JSON file."""
    if not os.path.exists(config_file):
        print(f"Warning: No config file found at {config_file}")
        return None

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            print(f"Loaded config from {config_file}")
            return config
    except Exception as e:
        print(f"Warning: Could not load config: {e}")
        return None


def create_sample_config(filename: str = "config.json") -> bool:
    """Create sample configuration file with all required fields."""
    if os.path.exists(filename):
        print(f"Config {filename} already exists")
        return False

    config = get_default_config()

    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Created config: {filename}")
    print(f"\nConfig includes:")
    print(f"  - DSPy model settings")
    print(f"  - LLM provider settings (for llm engine)")
    print(f"  - Default processing parameters")
    return True


async def async_main(args, config):
    """Async main processing function."""
    start_time = datetime.now()

    # Extract configurations
    model_config = config.get("model", {}) if config else {}
    defaults_config = config.get("defaults", {}) if config else {}

    # Get final values (CLI > config > defaults)
    source = args.source
    model_name = args.model or model_config.get("name")
    samples = args.samples or defaults_config.get("samples")
    mode = args.mode or defaults_config.get("mode")
    storage_mode = args.storage or defaults_config.get("storage")
    soap_engine = args.soap_engine or defaults_config.get("soap_engine")
    evaluation_mode = args.evaluation_mode or defaults_config.get(
        "evaluation_mode")
    output = args.output or defaults_config.get("output")
    batch_size = args.batch_size or defaults_config.get("batch_size", 10)

    # Check required parameters
    if not all([source, model_name, samples, mode, storage_mode, soap_engine, evaluation_mode, output]):
        print("Missing required parameters. Use --create-config or provide all via CLI")
        print("\nRequired parameters:")
        print(f"  --source: {source or 'MISSING'}")
        print(f"  --model: {model_name or 'MISSING'}")
        print(f"  --samples: {samples or 'MISSING'}")
        print(f"  --mode: {mode or 'MISSING'}")
        return 1

    # Validate model configuration for selected engine
    valid, error = validate_model_config(soap_engine, model_config)
    if not valid:
        print(f"Configuration error: {error}")
        return 1

    # Print configuration
    print(f"\n{'='*70}")
    print(f"DeepScribe SOAP Evaluation System")
    print(f"{'='*70}")
    print(f"Source:           {source}")
    print(f"Samples:          {samples}")
    print(f"Mode:             {mode}")
    print(f"SOAP Engine:      {soap_engine}")
    print(f"Evaluation Mode:  {evaluation_mode}")
    print(f"Batch Size:       {batch_size}")
    print(f"Output:           {output}")
    print(f"Storage Mode:     {storage_mode}")
    print(f"{'='*70}\n")

    # Create output directory if needed
    if os.path.dirname(output):
        os.makedirs(os.path.dirname(output), exist_ok=True)
        print(f"Output directory ready: {os.path.dirname(output)}")

    # Setup DSPy model (always needed for field detection and evaluation)
    print(f"\nSetting up DSPy model: {model_name}")
    if not setup_dspy_model(
        model_name,
        model_config.get("max_tokens", 4000),
        model_config.get("temperature", 0.1)
    ):
        return 1

    # Setup LLM client if using LLM engine
    llm_client = None
    prompt_file = None

    if soap_engine == "llm":
        print(f"\nSetting up LLM client for SOAP generation...")
        provider = model_config.get("provider", "openai")
        prompt_file = model_config.get(
            "prompt_file", "config/llm_prompts.yaml")

        llm_client = create_llm_client(
            provider=provider,
            model_name=model_name,
            temperature=model_config.get("temperature", 0.1),
            max_tokens=model_config.get("max_tokens", 4000)
        )

        if llm_client is None:
            print(f"Failed to create LLM client")
            return 1

        print(f"LLM client ready: {provider}/{model_name}")
        print(f"Using prompts from: {prompt_file}")

    print()

    # Initialize components
    print(f"Initializing data loader...")
    detector = DSPyFieldDetector()
    loader = UniversalDataLoader(detector)

    try:
        # Load and normalize data
        print(f"Loading data from: {source}")
        print(f"Requesting {samples} samples...")

        normalized_data, field_mapping = await loader.load_and_normalize(
            source=source,
            max_samples=samples
        )

        print(f"Loaded {len(normalized_data)} samples")
        print(f"Field mapping detected:")
        print(f"   - Transcript field: {field_mapping.transcript_field}")
        print(
            f"   - Reference notes field: {field_mapping.reference_notes_field}")
        print(
            f"   - Ground truth field: {field_mapping.ground_truth_field or 'None detected'}")
        if field_mapping.patient_metadata_fields:
            print(
                f"   - Metadata fields: {', '.join(field_mapping.patient_metadata_fields[:6])}")
        print(f"   - Confidence: {field_mapping.confidence_score:.2f}")
        print()

        # Initialize integration pipeline
        print(f"Initializing SOAP pipeline...")

        # Build engine kwargs based on selected engine
        engine_kwargs = {}
        if soap_engine == "llm":
            engine_kwargs['llm_client'] = llm_client
            engine_kwargs['model_name'] = model_name
            engine_kwargs['prompt_file'] = prompt_file

        integration = SimpleSOAPIntegration(
            soap_engine=soap_engine,
            evaluation_mode=evaluation_mode,
            storage_mode=storage_mode,
            storage_file=output,
            batch_size=batch_size,
            **engine_kwargs
        )
        print(f"Pipeline ready")
        print()

        # Process based on mode
        results = []

        if mode == "generate":
            print(f"Starting SOAP generation (batches of {batch_size})...")
            print(
                f"Total batches: {(len(normalized_data) + batch_size - 1) // batch_size}")
            print()
            results = await integration.process_normalized_data_async(normalized_data, source)

        elif mode == "evaluate":
            print(f"Starting evaluation (batches of {batch_size})...")
            print(
                f"Total batches: {(len(normalized_data) + batch_size - 1) // batch_size}")
            print()
            results = await integration.process_evaluation_only_async(normalized_data, source)

        else:  # both
            print(
                f"Starting generation + evaluation (batches of {batch_size})...")
            print(
                f"Total batches: {(len(normalized_data) + batch_size - 1) // batch_size}")
            print()
            results = await integration.process_normalized_data_async(normalized_data, source)

        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Print summary
        stats = integration.get_stats()
        print(f"\n{'='*70}")
        print(f"Processing Complete")
        print(f"{'='*70}")
        print(f"Processed samples:        {len(results)}")
        print(f"Total stored:             {stats.get('total_results', 0)}")
        print(f"Output file:              {output}")
        print(
            f"Processing time:          {duration:.1f}s ({duration/60:.1f}m)")
        print(
            f"Avg time per sample:      {duration/len(results):.2f}s" if results else "N/A")

        # Calculate success rate
        successful = len([r for r in results if 'error' not in r])
        if results:
            success_rate = (successful / len(results)) * 100
            print(
                f"Success rate:             {success_rate:.1f}% ({successful}/{len(results)})")

        print(f"{'='*70}\n")

        # Auto-generate dashboard if results exist
        if args.auto_dashboard and os.path.exists(output):
            print(f"Auto-generating dashboard...")
            try:
                from utils.dashboard import create_dashboard_from_files
                created_files = create_dashboard_from_files(
                    [output],
                    os.path.dirname(output) or "results",
                    f"SOAP Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )

                if 'dashboard' in created_files:
                    print(f"Dashboard created: {created_files['dashboard']}")

                    if args.open_dashboard:
                        import webbrowser
                        try:
                            webbrowser.open(
                                f"file://{os.path.abspath(created_files['dashboard'])}")
                            print(f"Dashboard opened in browser")
                        except:
                            pass
            except Exception as e:
                logger.warning(f"Dashboard auto-generation failed: {e}")

        return 0

    except Exception as e:
        logger.exception("Processing failed")
        print(f"\n{'='*70}")
        print(f"Error occurred: {e}")
        print(f"{'='*70}\n")
        return 1


def main():
    """Main entry point with argument parsing."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="DeepScribe SOAP Evaluation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create config file
  python main.py --create-config

  # Process samples with auto dashboard
  python main.py --source data.json --samples 10 --mode both --auto-dashboard

  # Create dashboard from existing results
  python main.py --dashboard results/file1.jsonl results/file2.jsonl

  # Create and open dashboard
  python main.py --dashboard results/*.jsonl --open

  # Process with LLM engine
  python main.py --source data.json --samples 10 --soap-engine llm
        """
    )

    # Required/main arguments
    parser.add_argument(
        "--source", help="Data source (file path or HuggingFace dataset)")
    parser.add_argument("--samples", type=int,
                        help="Number of samples to process")
    parser.add_argument("--mode", choices=VALID_OPTIONS["mode"],
                        help="Processing mode: generate, evaluate, or both")

    # Model and engine arguments
    parser.add_argument(
        "--model", help="Model name (e.g., gemini/gemini-2.5-pro, gpt-4)")
    parser.add_argument("--soap-engine", choices=VALID_OPTIONS["soap_engine"],
                        help="SOAP generation engine (dspy or llm)")
    parser.add_argument("--evaluation-mode", choices=VALID_OPTIONS["evaluation_mode"],
                        help="Evaluation mode: deterministic, llm_only, comprehensive, or skip")

    # Storage and output arguments
    parser.add_argument("--output", help="Output file path for results")
    parser.add_argument("--storage", choices=VALID_OPTIONS["storage"],
                        help="Storage mode: soap_only, evaluation_only, or both")

    # Performance arguments
    parser.add_argument("--batch-size", type=int,
                        help="Batch size for processing (default: 10)")

    # Config file arguments
    parser.add_argument("--config", default="config.json",
                        help="Config file path (default: config.json)")
    parser.add_argument("--create-config", action="store_true",
                        help="Create sample config file and exit")

    # Dashboard arguments
    parser.add_argument("--dashboard", nargs='*', metavar='FILE',
                        help="Create dashboard from result files (specify files or use --output)")
    parser.add_argument("--dashboard-title", default="SOAP Evaluation Dashboard",
                        help="Title for the dashboard")
    parser.add_argument("--auto-dashboard", action="store_true",
                        help="Auto-generate dashboard after processing")
    parser.add_argument("--open-dashboard", "--open", action="store_true",
                        help="Automatically open dashboard in browser")

    args = parser.parse_args()

    # Handle config creation
    if args.create_config:
        return 0 if create_sample_config(args.config) else 1

    # Handle dashboard creation
    if args.dashboard is not None:
        from utils.dashboard import create_dashboard_from_files

        # Determine which files to use
        if len(args.dashboard) == 0:
            # Use current output file or default
            if args.output and os.path.exists(args.output):
                dashboard_files = [args.output]
            else:
                # Look for any result files in results directory
                results_dir = "results"
                if os.path.exists(results_dir):
                    dashboard_files = [
                        os.path.join(results_dir, f)
                        for f in os.listdir(results_dir)
                        if f.endswith('.jsonl') or f.endswith('.json')
                    ]
                    if not dashboard_files:
                        print("No result files found in results/ directory")
                        return 1
                else:
                    print("No output file specified and no results/ directory found")
                    print(
                        "Usage: python main.py --dashboard <file1.jsonl> [file2.jsonl ...]")
                    return 1
        else:
            # Use specified files
            dashboard_files = args.dashboard
            # Verify files exist
            missing_files = [
                f for f in dashboard_files if not os.path.exists(f)]
            if missing_files:
                print(f"Error: Files not found: {', '.join(missing_files)}")
                return 1

        print(f"\nCreating dashboard from {len(dashboard_files)} file(s)...")
        for f in dashboard_files:
            print(f"  - {f}")
        print()

        try:
            created_files = create_dashboard_from_files(
                dashboard_files,
                "results",
                args.dashboard_title
            )

            print("Dashboard created successfully:")
            for file_type, file_path in created_files.items():
                print(f"  - {file_type.title()}: {file_path}")
            print()

            # Open dashboard in browser if requested
            if args.open_dashboard and 'dashboard' in created_files:
                import webbrowser
                try:
                    dashboard_path = os.path.abspath(
                        created_files['dashboard'])
                    webbrowser.open(f"file://{dashboard_path}")
                    print(f"Dashboard opened in browser")
                except Exception as e:
                    print(f"Could not open browser automatically: {e}")
                    print(
                        f"Open manually: file://{os.path.abspath(created_files['dashboard'])}")

            return 0

        except Exception as e:
            logger.exception("Dashboard creation failed")
            print(f"Dashboard creation failed: {e}")
            return 1

    # Load config for normal processing
    config = load_config(args.config)

    # Run async main
    try:
        return asyncio.run(async_main(args, config))
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        logger.exception("Fatal error")
        print(f"\nFatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
