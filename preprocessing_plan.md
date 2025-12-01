⭐ Overview

This document defines every preprocessing technique that our platform will support.
It unifies preprocessing for:

Classical ML (tabular)

NLP fine-tuning (LLM training datasets)

Full robustness & fail-safe guarantees

Our goal:

Create the strongest, safest, most flexible preprocessing engine available on the web.

The list below contains 131 operations grouped into categories.
Each operation is modular, composable, and can be applied per-column or globally.

```
1️⃣ Classical ML (Tabular) Preprocessing

These apply to CSV, Excel, Parquet, TSV and general structured datasets.

A. Missing Value Handling (10 items)

Mean imputation

Median imputation

Mode imputation

Constant fill (user-defined)

Drop rows containing missing values

Drop columns with missing values

Forward fill

Backward fill

Missing-indicator column (flag NaNs)

Model-based imputation (advanced)

B. Data Type Fixing (6 items)

Cast to integer

Cast to float

Cast to categorical

Parse to datetime

Convert numeric-like strings (e.g., “12,000”, “10k”)

Auto-detect binary columns

C. Encoding (Categorical → Numeric) (8 items)

One-hot encoding

Label encoding

Ordinal encoding

Target encoding

K-fold target encoding

Hashing encoding

Binary encoding

Leave-one-out encoding

D. Scaling (Numeric) (7 items)

StandardScaler (z-score)

MinMaxScaler

RobustScaler

Normalizer (L1/L2)

Log scaling

Outlier clipping

Quantile transformer

E. Feature Engineering (9 items)

Polynomial features

Interaction features

Bucketization / Binning (equal width or quantile)

Date feature splits (year/month/day/week/hour)

Text length features (num_chars, num_words)

Frequency encoding

Group-by aggregate features

Time-series rolling aggregates

Lag features

F. Feature Selection (7 items)

Select K-best

Variance threshold

Correlation-based dropping

Remove multicollinearity (VIF)

Feature importance selection (model-based)

Manual column selection

Auto-detect redundant fields (IDs, UUIDs)

G. Row-Level Operations (4 items)

Deduplication

Conditional filtering

Outlier removal / Winsorization

Aggregation-based reduction

H. Data Cleaning / Sanitization (8 items)

Trim whitespace

Lowercase/uppercase normalization

Remove emojis

Remove special characters

Regex pattern replacements

Remove HTML tags

Unicode normalization

Fix encoding errors

2️⃣ NLP / Fine-Tuning Preprocessing

For datasets used in LLM fine-tuning, supervised chat models, or instruction datasets.

A. Text Cleaning (9 items)

Lowercase text

Uppercase text

Remove punctuation

Remove stopwords

Lemmatization

Stemming

Remove emojis

Remove URLs

Strip HTML tags

B. Tokenization-Aware Processing (7 items)

Token count estimation

Max-token truncation

Token overflow warnings

Word-piece/BPE simulation

Sentencepiece-based splitting

Padding (if required)

Token normalization

C. Prompt/Completion Construction (7 items)

Template-based prompt generation

Template-based completion generation

Multi-column → prompt merge

System prefix injection

Role-based formatting (user/assistant)

Chat-style conversation formatting

Reinforcement prompt formatting

D. Chunking (6 items)

Chunk by token count

Sliding window chunking (stride overlap)

Sentence-boundary chunking

Paragraph-level chunking

Hard truncation

Smart truncation (sentence-first)

E. Deduplication (4 items)

Exact duplicate removal

Near-duplicate removal (fuzzy)

Deduplicate by target column

Deduplicate Q/A templates

F. PII Removal / Safety (8 items)

Email redaction

Phone number redaction

Address redaction

ID masking (UUID, Aadhar-like patterns)

Profanity removal

Regex custom PII removal

Strict PII mode (GDPR-ready)

IP address masking

G. JSONL Preparation (7 items)

CSV → JSONL

JSON → JSONL

Multi-column → prompt+completion format

Escape unsafe characters

Validate each JSONL line

Base64-safe encoding

Cleaning invalid UTF-8

H. Dataset Splitting (4 items)

Train/Validation/Test split

Stratified split

Seed-based deterministic splits

Shuffle/no-shuffle modes

I. Metadata Generation (6 items)

Tokenizer name

Base model name

Training sample count

Average tokens per example

Safety warnings

Recommended training LR / batch size

3️⃣ Robustness & Fail-Proof Processing

Ensures no crashes, no memory overflow, no corrupted outputs.

J. Robust Parsing (6 items)

Chunked CSV reading

on_bad_lines="skip" with error capture

Encoding detection fallback

Corrupted CSV recovery

Streaming processing for huge files

64-bit safe row counting

K. System-Level Safeguards (6 items)

File size checks

Memory guard rails

CPU throttling

Timeout for slow operations

Kill-switch for runaway tasks

Disk usage checks

L. Artifact Safety (5 items)

Atomic writes (.partial → final rename)

Temporary backups for each step

Gzip archived versions

Restore last stable version on failure

Metadata about each transformation applied
```

🌟 Summary Table
```
Category	Count
Classical ML Preprocessing	59
NLP / Fine-Tuning Preprocessing	55
Robustness & Fail-Safes	17
TOTAL	131 techniques
```

🚀 How the Preprocessing Engine Works (High-Level Approach)
```
User selects a mode

"tabular"

"nlp_finetune"

Frontend sends a JSON config specifying:

Columns

Operations

Order of execution

Parameters per transformation

Backend runs a DAG-style preprocessing pipeline

Each step is atomic

All steps are reversible

Each produces logs + metadata

On completion, backend returns:

Processed file (CSV or JSONL)

Preview

Metadata

Token counts (for NLP)

Safety warnings

Suggestions

If any step fails

Rollback

Restore previous artifact

Return safe error message
```
🎯 Why This Matters

This architecture allows:

Both beginners and advanced ML developers to customize preprocessing.

Fine-grained control on column-level transformations.

Maximum safety — no crashes due to bad datasets.

Support for massive file sizes.

Seamless integration with classical ML training + LLM fine-tuning.