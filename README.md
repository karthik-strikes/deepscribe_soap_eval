# DeepScribe AI Evaluation Suite

**A working evaluation system built with DSPy and LLMs for medical SOAP note quality assessment.**

---

## 🛠️ **What I Built**

### **Core Evaluation System**

I developed a comprehensive evaluation framework using:

- **DSPy Framework**: Built structured LLM evaluators with `dspy.ChainOfThought` modules
- **Multi-Model Support**: Integrated Google Gemini, OpenAI GPT, and Anthropic Claude
- **Async Processing**: Implemented true batch processing with `asyncio` for production speed
- **Interactive Analytics**: Created real-time dashboards with Plotly for quality monitoring

### **Technical Implementation**

#### **1. LLM-Based Evaluators (Built with DSPy)**

```python
# ContentFidelityEvaluator - Detects missing findings & hallucinations
class ContentFidelityEvaluator(dspy.Module):
    def __init__(self):
        self.extract_ground_truth = dspy.ChainOfThought(ExtractCriticalFindings)
        self.validate_content = dspy.ChainOfThought(ValidateContentFidelity)
    
    # Two-stage evaluation process:
    # 1. Extract critical medical facts from transcript
    # 2. Check what's captured vs missed vs hallucinated in SOAP note
```

#### **2. Medical Accuracy Validation**

```python
# MedicalCorrectnessEvaluator - Validates clinical accuracy
class MedicalCorrectnessEvaluator(dspy.Module):
    def __init__(self):
        self.extract_statements = dspy.ChainOfThought(ExtractMedicalStatements)
        self.validate_accuracy = dspy.ChainOfThought(ValidateMedicalAccuracy)
    
    # Evaluates medical correctness of diagnoses, treatments, recommendations
```

#### **3. Fast Deterministic Evaluators**

```python
# Built regex-based evaluators for speed
class EntityCoverageEvaluator:
    medical_patterns = {
        'medications': r'\b(?:\w+(?:cillin|mycin|pril)|mg|tablet)\b',
        'symptoms': r'\b(?:pain|fever|nausea|headache|dizzy)\b',
        'vital_signs': r'\b(?:\d{2,3}/\d{2,3}|\d{2,3}\s*bpm)\b'
    }
    # Matches medical entities between transcript and SOAP note
```

#### **4. Production Pipeline Architecture**

```python
# SimpleSOAPIntegration - Orchestrates everything
class SimpleSOAPIntegration:
    def __init__(self, soap_engine="dspy", evaluation_mode="comprehensive"):
        self.soap_pipeline = SOAPGenerationPipeline(engine_type=soap_engine)
        self.evaluator = EvaluationPipeline()  # Coordinates all evaluators
        self.storage = AsyncStorageWrapper()   # Handles results storage
    
    # Processes batches of conversations -> SOAP notes -> evaluations
```

---

## 🔬 **Approach & Design Decisions**

### **Evaluation Approach Comparison**

#### **1. LLM-as-Judge vs Deterministic Metrics**

I implemented a **hybrid approach** after considering the tradeoffs:

**LLM-as-Judge (ContentFidelityEvaluator, MedicalCorrectnessEvaluator):**

- ✅ **Pros**: Deep semantic understanding, catches nuanced medical errors, handles context
- ❌ **Cons**: Slower (~8s per evaluation), expensive API calls, potential inconsistency
- **Use Case**: Production quality assessment, final validation

**Deterministic Metrics (EntityCoverageEvaluator, SOAPCompletenessEvaluator):**

- ✅ **Pros**: Lightning fast (~0.1s), consistent, no API costs, reliable baselines
- ❌ **Cons**: Surface-level analysis, misses context, limited medical knowledge
- **Use Case**: Quick CI/CD checks, large-scale screening

**Why Hybrid**: Combines speed of deterministic with depth of LLM analysis for optimal production workflow.

#### **2. Reference-Based vs Non-Reference Evaluation**

I built **intelligent fallback logic**:

```python
# Smart comparison strategy I implemented
if ground_truth and ground_truth != reference_soap:
    eval_source = ground_truth  # Use gold standard when available
    compared_on = "ground_truth"
else:
    eval_source = transcript    # Fallback to transcript analysis
    compared_on = "transcript"
```

**Reference-Based (Ground Truth):**

- ✅ **Pros**: Objective comparison, clear quality standards, reliable metrics
- ❌ **Cons**: Expensive to curate, limited scalability, not always available
- **When I Use It**: When ground truth SOAP notes exist in dataset

**Non-Reference (Transcript-Based):**

- ✅ **Pros**: Scalable, works with any conversation, no manual annotation needed
- ❌ **Cons**: Subjective quality assessment, harder to establish clear standards
- **When I Use It**: Production monitoring, new conversations without ground truth

#### **3. Synchronous vs Asynchronous Processing**

I chose **true async batch processing**:

**Why Not Sequential Processing:**

- Would take 80s for 10 evaluations (8s each)
- Blocks on API calls, poor resource utilization
- Doesn't scale for production volumes

**Why Async Batch Processing:**

```python
# True parallel evaluation - 3-5x speedup
eval_results = await asyncio.gather([
    evaluator.evaluate_batch_async(transcripts, notes, metadata)
    for evaluator in self.evaluators
])
```

- Processes 10+ notes simultaneously
- Better API utilization, faster feedback
- Scales to production workloads

#### **4. DSPy vs Raw Prompting**

I chose **DSPy framework** over raw prompting:

**Raw Prompting Approach:**

- Manual prompt engineering, harder to optimize
- No structured outputs, JSON parsing issues
- Difficult to maintain consistency across evaluators

**DSPy Structured Approach:**

```python
class ExtractCriticalFindings(dspy.Signature):
    transcript: str = dspy.InputField(desc="Patient conversation")
    critical_findings: str = dspy.OutputField(desc="JSON list of critical facts")
```

- Structured inputs/outputs with type safety
- Built-in optimization capabilities (BootstrapFewShot)
- Consistent evaluation patterns across different medical domains

#### **5. Multi-Model Strategy**

I implemented **provider flexibility**:

**Single Model Risk**: Vendor lock-in, API outages, model deprecation
**Multi-Model Benefits**:

- Easy switching between Gemini, GPT-4, Claude
- Cost optimization (Gemini cheaper, GPT-4 more accurate)
- Redundancy for production reliability

### **Key Architecture Decisions**

#### **Evaluation Modes**

```python
# Flexible evaluation for different use cases
evaluation_modes = {
    "deterministic": "Fast baseline (~2s per sample)",
    "llm_only": "Deep analysis (~8s per sample)", 
    "comprehensive": "Best quality (~10s per sample)"
}
```

#### **Storage Strategy**

- **JSONL Format**: Streaming for large datasets, easy to append
- **Structured Results**: Consistent schema for downstream analysis
- **Dashboard Integration**: Automatic visualization generation

#### **Error Handling Philosophy**

```python
# Graceful degradation - continue processing even if some evaluations fail
try:
    eval_result = await evaluator.evaluate_async(transcript, note)
except Exception as e:
    eval_result = create_fallback_result(error=str(e))
```

### **Tradeoffs Made**

1. **Accuracy vs Speed**: Hybrid approach provides configurable balance
2. **Cost vs Quality**: Deterministic fallbacks reduce API costs while maintaining quality floor
3. **Complexity vs Flexibility**: Modular architecture enables easy extension but requires more setup
4. **Memory vs Processing**: Streaming JSONL allows large datasets but requires careful batch sizing

This design prioritizes **production scalability** while maintaining **research-quality insights** - exactly what medical AI systems need for both development and deployment.

---

## 🚀 **How It Actually Works**

When you run:

```bash
python main.py --source "adesouza1/soap_notes" --samples 10 --auto-dashboard
```

**Here's what happens under the hood:**

### **Step 1: Data Loading & Model Setup**

```
🔄 Loading HuggingFace dataset: adesouza1/soap_notes
📥 Downloaded conversations and ground truth SOAP notes
🤖 Initializing DSPy with Gemini-2.5-Pro (or your configured model)
⚡ Setting up async batch processing pipeline
```

### **Step 2: SOAP Note Generation**

```python
# Your system generates SOAP notes using DSPy structured generation
soap_result = await self.soap_pipeline.forward_async(conversation, metadata)

# Output structure:
{
    "subjective": "Patient reports chest pain...",
    "objective": "Vital signs: BP 140/90...", 
    "assessment": "Primary diagnosis: Acute coronary syndrome...",
    "plan": "Order EKG, start aspirin..."
}
```

### **Step 3: Multi-Layer Evaluation**

```python
# Runs multiple evaluators in parallel
deterministic_results = [
    entity_coverage.evaluate(transcript, soap_note),      # ~0.1s
    soap_completeness.evaluate(soap_note),                # ~0.1s  
    format_validity.evaluate(soap_note)                   # ~0.1s
]

llm_results = await asyncio.gather([
    content_fidelity.evaluate_async(transcript, soap_note),    # ~8s
    medical_correctness.evaluate_async(transcript, soap_note)  # ~8s  
])
```

### **Step 4: Real Results Generated**

```json
{
  "conversation": "Doctor: How are you feeling today?...",
  "generated_soap": "SUBJECTIVE: Patient reports...",
  "evaluation_metrics": {
    "overall_quality": 87.3,
    "content_fidelity_f1": 0.82,
    "medical_accuracy": 0.91,
    "entity_coverage": 85.0,
    "section_completeness": 100.0,
    "correctly_captured": ["chest pain", "shortness of breath"],
    "missed_critical": ["family history"],
    "hallucinations": []
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### **Step 5: Interactive Dashboard Creation**

```python
# utils/dashboard.py automatically generates:
dashboard = SOAPEvaluationDashboard(results_files)
dashboard.create_comprehensive_dashboard("results/dashboard.html")

# Creates interactive Plotly charts showing:
# - Quality trends over time
# - Score distributions  
# - Issue breakdowns
# - Performance metrics
```

---

## 🎯 **Real Usage Examples**

### **Quick Evaluation**

```bash
# Processes 5 conversations, generates SOAP notes, runs evaluation
python main.py --source "adesouza1/soap_notes" --samples 5

# What happens:
# 1. Downloads dataset from HuggingFace
# 2. Initializes Gemini model via DSPy
# 3. Generates 5 SOAP notes (parallel processing)
# 4. Runs 5 evaluators on each note
# 5. Saves results to results/soap_results.jsonl
# 6. Processing time: ~2-3 minutes
```

### **Production Monitoring**

```bash
# Monitor quality of existing SOAP notes
python main.py --source "production_notes.json" --mode evaluate --samples 100

# Evaluates 100 existing notes for:
# - Missing critical information
# - Hallucinated content  
# - Medical accuracy issues
# - Generates quality trends dashboard
```

### **Model Comparison**

```bash
# Compare two different models
python main.py --source data.csv --model "openai/gpt-4o-mini" --output results/gpt4_results.jsonl
python main.py --source data.csv --model "gemini/gemini-2.5-pro" --output results/gemini_results.jsonl

# Create comparison dashboard
python main.py --dashboard results/gpt4_results.jsonl results/gemini_results.jsonl
```

---

## 🎛️ **Complete CLI Reference**

### **Required Arguments**

#### **`--source`** (Required)

```bash
# HuggingFace dataset
python main.py --source "adesouza1/soap_notes"

# Local CSV file (auto-detects columns: conversation, transcript, soap_note, etc.)
python main.py --source "medical_data.csv"

# Local JSON/JSONL file
python main.py --source "conversations.json"
```

**What it does**: Specifies where to load conversation data from. System auto-detects format and extracts relevant fields.

### **Core Processing Arguments**

#### **`--samples`** (Integer)

```bash
# Process 5 conversations
python main.py --source "adesouza1/soap_notes" --samples 5

# Process 100 for production analysis
python main.py --source data.csv --samples 100
```

**What it does**: Limits how many conversations to process. Useful for testing or resource management.
**Default**: From config.json (usually 10)

#### **`--mode`** (Choices: `generate`, `evaluate`, `both`)

```bash
# Generate SOAP notes only (no evaluation)
python main.py --source data.csv --mode generate

# Evaluate existing SOAP notes only
python main.py --source notes.json --mode evaluate

# Generate + evaluate (default)
python main.py --source data.csv --mode both
```

**What it does**:

- `generate`: Creates SOAP notes from conversations using DSPy
- `evaluate`: Runs evaluation on existing SOAP notes
- `both`: Full pipeline - generate then evaluate

#### **`--evaluation-mode`** (Choices: `deterministic`, `llm_only`, `comprehensive`, `skip`)

```bash
# Fast baseline checks only (~2s per sample)
python main.py --source data.csv --evaluation-mode deterministic

# Deep LLM analysis only (~8s per sample)
python main.py --source data.csv --evaluation-mode llm_only

# Best quality - all evaluators (~10s per sample)
python main.py --source data.csv --evaluation-mode comprehensive

# Skip evaluation entirely (generation only)
python main.py --source data.csv --evaluation-mode skip
```

**What it does**:

- `deterministic`: EntityCoverageEvaluator, SOAPCompletenessEvaluator, FormatValidityEvaluator only
- `llm_only`: ContentFidelityEvaluator, MedicalCorrectnessEvaluator only  
- `comprehensive`: All 5 evaluators for complete analysis
- `skip`: No evaluation (same as `--mode generate`)

### **Model & Engine Configuration**

#### **`--model`** (String)

```bash
# Use Google Gemini (fast, cost-effective)
python main.py --source data.csv --model "gemini/gemini-2.5-pro"

# Use OpenAI GPT (more accurate)
python main.py --source data.csv --model "openai/gpt-4o-mini"

# Use Anthropic Claude
python main.py --source data.csv --model "anthropic/claude-3-5-sonnet-20241022"
```

**What it does**: Specifies which LLM to use for SOAP generation and evaluation. Requires corresponding API key.
**Default**: From config.json (usually gemini/gemini-2.5-pro)

#### **`--soap-engine`** (Choices: `dspy`)

```bash
# Use DSPy structured generation (recommended)
python main.py --source data.csv --soap-engine dspy
```

**What it does**: DSPy engine uses structured generation with ChainOfThought modules for consistent, high-quality SOAP notes.
**Default**: `dspy`

### **Output & Storage**

#### **`--output`** (File path)

```bash
# Custom output file
python main.py --source data.csv --output "results/my_evaluation.jsonl"

# Output to specific directory
python main.py --source data.csv --output "/path/to/results.jsonl"
```

**What it does**: Specifies where to save evaluation results in JSONL format.
**Default**: `results/soap_results.jsonl`

#### **`--storage`** (Choices: `soap_only`, `evaluation_only`, `both`)

```bash
# Store only SOAP notes (no evaluation metrics)
python main.py --source data.csv --storage soap_only

# Store only evaluation results (no SOAP content)
python main.py --source data.csv --storage evaluation_only

# Store everything (default)
python main.py --source data.csv --storage both
```

**What it does**: Controls what data gets saved to reduce file size or focus on specific outputs.

### **Performance Tuning**

#### **`--batch-size`** (Integer)

```bash
# Small batches for limited memory
python main.py --source data.csv --batch-size 5

# Large batches for faster processing
python main.py --source data.csv --batch-size 20
```

**What it does**: Number of samples processed in parallel. Higher = faster but more memory usage.
**Default**: 10 (good balance for most systems)

### **Configuration Management**

#### **`--config`** (File path)

```bash
# Use custom config file
python main.py --source data.csv --config "my_config.json"

# Use default config
python main.py --source data.csv --config "config.json"
```

**What it does**: Specifies which configuration file to load model settings and defaults from.
**Default**: `config.json`

#### **`--create-config`** (Flag)

```bash
# Create a sample configuration file
python main.py --create-config

# Create config with custom name
python main.py --create-config --config "production_config.json"
```

**What it does**: Generates a template configuration file with all available options and exits.

### **Dashboard & Analytics**

#### **`--auto-dashboard`** (Flag)

```bash
# Auto-generate dashboard after processing
python main.py --source data.csv --samples 10 --auto-dashboard
```

**What it does**: Automatically creates interactive HTML dashboard from evaluation results when processing completes.

#### **`--dashboard`** (File list)

```bash
# Create dashboard from specific files
python main.py --dashboard results/file1.jsonl results/file2.jsonl

# Create dashboard from all results in directory
python main.py --dashboard

# Create dashboard from current output
python main.py --source data.csv --output results/test.jsonl --dashboard
```

**What it does**: Creates interactive dashboard from existing evaluation result files. Can specify files or auto-detect.

#### **`--dashboard-title`** (String)

```bash
# Custom dashboard title
python main.py --dashboard results/*.jsonl --dashboard-title "Model Comparison Analysis"
```

**What it does**: Sets custom title for generated dashboard.
**Default**: "SOAP Evaluation Dashboard"

#### **`--open-dashboard` / `--open`** (Flag)

```bash
# Create dashboard and open in browser
python main.py --dashboard results/*.jsonl --open
```

**What it does**: Automatically opens the generated dashboard in your default web browser.

### **Complete Example Commands**

#### **Quick Development Test**

```bash
python main.py --source "adesouza1/soap_notes" --samples 5 --evaluation-mode deterministic --auto-dashboard --open
```

**Result**: Downloads dataset, processes 5 samples with fast evaluation, creates dashboard, opens in browser (~30 seconds)

#### **Production Quality Analysis**

```bash
python main.py --source "production_notes.json" --mode evaluate --evaluation-mode comprehensive --samples 100 --batch-size 10 --output "results/prod_analysis.jsonl" --auto-dashboard
```

**Result**: Evaluates 100 existing notes with all evaluators, saves detailed results, generates dashboard (~15 minutes)

#### **Model Comparison Workflow**

```bash
# Test Gemini
python main.py --source data.csv --model "gemini/gemini-2.5-pro" --samples 20 --output "results/gemini_results.jsonl"

# Test GPT-4
python main.py --source data.csv --model "openai/gpt-4o-mini" --samples 20 --output "results/gpt4_results.jsonl"

# Compare results
python main.py --dashboard results/gemini_results.jsonl results/gpt4_results.jsonl --dashboard-title "Gemini vs GPT-4 Comparison" --open
```

**Result**: Side-by-side model performance comparison with interactive dashboard

#### **Configuration Setup**

   ```bash
# Create initial config
python main.py --create-config

# Edit config.json with your preferences, then run
python main.py --source "adesouza1/soap_notes" --samples 10
```

**Result**: Persistent configuration for consistent runs

---

## ⚙️ **Configuration & Setup**

### **1. Install Dependencies**

   ```bash
pip install -r requirements.txt
# Installs: dspy-ai, datasets, pandas, plotly, asyncio libraries
   ```

### **2. API Keys**

   ```bash
# Set your model API key
export GEMINI_API_KEY="your-actual-api-key"
export OPENAI_API_KEY="your-openai-key"  
export ANTHROPIC_API_KEY="your-anthropic-key"
```

### **3. Model Configuration** (`config.json`)

```json
{
  "model": {
    "name": "gemini/gemini-2.5-pro",    // Which LLM to use
    "max_tokens": 4000,                 // Response length limit
    "temperature": 0.1                  // Deterministic outputs
  },
  "defaults": {
    "samples": 10,                      // How many to process
    "evaluation_mode": "comprehensive", // All evaluators
    "batch_size": 10                    // Parallel processing count
  }
}
```

---

## 🔧 **Technical Architecture**

### **File Structure I Built**

```
deepscribe_soap_eval/
├── main.py                    # CLI interface with argparse
├── config.json               # Model and processing configuration
├── requirements.txt          # Python dependencies
├── core/
│   ├── soap_generator.py     # DSPy-based SOAP generation
│   ├── integration.py        # Async pipeline orchestration  
│   └── storage.py           # JSONL results storage
├── evaluation/
│   └── evaluator.py         # All evaluation logic (5 evaluators)
├── utils/
│   ├── dashboard.py         # Plotly dashboard generation
│   ├── model_setup.py       # LLM client configuration
│   └── json_parser.py       # Robust JSON parsing for LLM outputs
├── data/
│   └── loader.py           # HuggingFace + CSV data loading
└── results/               # Auto-generated outputs
    ├── dashboard.html        # Interactive quality dashboard
    ├── quality_report.html   # Statistical analysis
    └── *.jsonl              # Evaluation results
```

### **Processing Pipeline**

```
Data Input → Model Setup → SOAP Generation → Evaluation → Dashboard
     ↓            ↓              ↓             ↓           ↓
HuggingFace → DSPy Init → ChainOfThought → 5 Evaluators → Plotly
   CSV          Gemini      Async Batch     Parallel      HTML
   JSON         OpenAI        Processing    Processing   Interactive
```

---

## 📊 **What You Get**

### **Detailed Results**

Every evaluation produces structured output:

```json
{
  "original_transcript": "Full conversation...",
  "generated_soap_note": "SUBJECTIVE: ...\nOBJECTIVE: ...",
  "evaluation_metrics": {
    "deterministic_metrics": {
      "entity_coverage": 85.0,
      "section_completeness": 100.0, 
      "format_validity": 95.0
    },
    "llm_metrics": {
      "content_fidelity": {"f1": 0.82, "precision": 0.89, "recall": 0.76},
      "medical_correctness": {"accuracy": 0.91}
    }
  },
  "processing_time": "4.2s",
  "model_used": "gemini/gemini-2.5-pro"
}
```

### **Interactive Dashboard**

- **Quality Timeline**: Track scores over time with trend analysis
- **Distribution Charts**: Histogram of quality scores, grade distribution  
- **Issue Analysis**: Breakdown of missing findings, hallucinations, errors
- **Performance Metrics**: Processing speed, success rates, model comparison

### **Statistical Summary**

```json
{
  "total_samples": 50,
  "avg_quality": 87.3,
  "grade_distribution": {"A": 32, "B": 15, "C": 3},
  "success_rate": 96.0,
  "processing_speed": "4.2s per sample",
  "issues_found": {
    "missed_critical": 23,
    "hallucinations": 8, 
    "medical_errors": 5
  }
}
```

---

## 🚀 **Performance & Scale**

### **Speed Optimization**

- **Async Processing**: 3-5x faster than sequential evaluation
- **Batch Operations**: Process 10+ notes simultaneously
- **Smart Caching**: Avoid reprocessing duplicate conversations
- **Streaming Output**: Memory-efficient for large datasets

### **Scalability Features**

- **Configurable Batch Sizes**: Adjust based on available resources
- **Multiple Output Formats**: JSONL, JSON, dashboard HTML
- **Resume Capability**: Continue from previous runs
- **Error Handling**: Graceful degradation on API failures

### **Sample Execution Output**

See `results/` folder for actual execution logs. Here's what a typical run looks like:

```bash
$ python main.py --source "adesouza1/soap_notes" --samples 60 --batch-size 20 --mode both --evaluation-mode comprehensive

======================================================================
DeepScribe SOAP Evaluation System
======================================================================
Source:           adesouza1/soap_notes
Samples:          60
Mode:             both
SOAP Engine:      dspy
Evaluation Mode:  comprehensive
Batch Size:       20
Output:           results/generate_evaluate_comprehensive5.jsonl
======================================================================

Setting up DSPy model: gemini/gemini-2.5-pro
Loading data from: adesouza1/soap_notes
Loaded 60 samples
Field mapping detected:
   - Transcript field: patient_convo
   - Reference notes field: soap_notes
   - Ground truth field: soap_notes

Starting generation + evaluation (batches of 20)...
Total batches: 3

Processing progress: 100%|██████████| 60/60 [05:20<00:00, 5.34s/sample]

======================================================================
Processing Complete
======================================================================
Processed samples:        60
Total stored:             60
Output file:              results/generate_evaluate_comprehensive5.jsonl
Processing time:          322.4s (5.4m)
Avg time per sample:      5.37s
Success rate:             100.0% (60/60)
======================================================================
```

**What happened:**
- Downloaded HuggingFace dataset `adesouza1/soap_notes`
- Processed 60 conversations in 3 batches of 20
- Generated SOAP notes using DSPy structured approach
- Ran comprehensive evaluation (all 5 evaluators)
- Saved detailed results to JSONL file
- **Total time**: 5.4 minutes for 60 samples
- **Performance**: 5.37 seconds per sample average

---

## 🛠️ **Development Highlights**

### **What Makes This Production-Ready**

1. **Robust JSON Parsing**: Handles malformed LLM outputs with 3-tier fallback
2. **Async Architecture**: True parallel processing, not just concurrent
3. **Flexible Data Loading**: HuggingFace, CSV, JSON auto-detection
4. **Interactive Analytics**: Real-time dashboard generation
5. **Configuration Management**: Easy model switching and parameter tuning

### **Key Technical Decisions**

- **DSPy Framework**: Structured LLM interactions vs raw prompting
- **Hybrid Evaluation**: LLM accuracy + deterministic speed
- **Batch Processing**: True batching vs parallel singles for efficiency  
- **Streaming Storage**: JSONL for large-scale processing
- **Component Architecture**: Modular evaluators for extensibility

This system represents a complete, working solution for medical AI evaluation that can actually be deployed and used in production. 🎯
