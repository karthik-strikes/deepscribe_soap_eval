# DeepScribe SOAP Evaluation System

Simple system for generating and evaluating SOAP medical notes using DSPy and LLMs.

## 🚀 Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export GEMINI_API_KEY="your-api-key-here"
```

### 3. Run

```bash
python main.py --source "adesouza1/soap_notes"
```

## 📋 Usage

### Basic Commands

```bash
# Generate and evaluate SOAP notes (default)
python main.py --source "adesouza1/soap_notes" --samples 5

# Only generate (no evaluation)
python main.py --source "data.csv" --mode generate --samples 10

# Only evaluate existing notes
python main.py --source "data.json" --mode evaluate

# Custom output file
python main.py --source "data.csv" --output "my_results.jsonl"
```

### Arguments

- `--source` (required): Your data source
- `--samples`: Number of samples (default: 5)
- `--mode`: `generate`, `evaluate`, or `both` (default: both)
- `--output`: Output file (default: soap_results.jsonl)
- `--model`: LLM model to use

## ⚙️ Configuration

Edit `config.json` to change defaults:

```json
{
  "model": {
    "name": "gemini/gemini-2.5-pro",
    "max_tokens": 4000,
    "temperature": 0.1
  },
  "defaults": {
    "samples": 5,
    "mode": "both",
    "eval_notes": "generated",
    "output": "results/soap_results.jsonl"
  }
}
```

### Supported Models

- `gemini/gemini-2.5-pro` (needs GEMINI_API_KEY)
- `openai/gpt-4o-mini` (needs OPENAI_API_KEY)
- `anthropic/claude-3-5-sonnet-20241022` (needs ANTHROPIC_API_KEY)

## 📊 Input Data

### CSV Files

```bash
python main.py --source "medical_data.csv"
```

Your CSV should have columns like `conversation`, `transcript`, `soap_note`, etc.

### JSON Files

```bash
python main.py --source "conversations.json"
```

### HuggingFace Datasets

```bash
python main.py --source "adesouza1/soap_notes"
```

## 📈 Output

All results are automatically saved in the `results/` folder as JSONL files:

```bash
results/
├── soap_results.jsonl          # Default output file
├── my_custom_results.jsonl     # Custom output files
└── ...
```

Example result format:

```json
{
  "original_transcript": "Doctor: Hello...",
  "generated_soap_note": "SUBJECTIVE: Patient reports...",
  "evaluation_metrics": {"overall_quality_score": 85.0},
  "timestamp": "2024-01-15T10:30:00"
}
```

## 🔧 Examples

```bash
# Quick test
python main.py --source "adesouza1/soap_notes" --samples 3

# Generate 20 notes, no evaluation
python main.py --source "data.csv" --mode generate --samples 20

# Use different model
python main.py --source "data.json" --model "openai/gpt-4o-mini"

# Custom config file
python main.py --source "data.csv" --config "my_config.json"
```

## 🛠️ Troubleshooting

1. **API Key Error**: Make sure your API key is set

   ```bash
   echo $GEMINI_API_KEY
   ```

2. **File Not Found**: Use full path

   ```bash
   python main.py --source "/full/path/to/data.csv"
   ```

3. **Memory Issues**: Reduce samples

   ```bash
   python main.py --source "data" --samples 2
   ```

## 📁 Files

```
deepscribe_soap_eval/
├── main.py           # Main script
├── config.json       # Configuration
├── requirements.txt  # Dependencies
├── README.md         # This file
├── results/          # All output files go here
├── core/            # Core components
├── data/            # Data loading
└── evaluation/      # Evaluation logic
```

That's it! Simple and clean. 🎯
