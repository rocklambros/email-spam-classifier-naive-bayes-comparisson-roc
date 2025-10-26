# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic project for a Master of Science in Applied Data Science and AI program, specifically for the "Essential Math for Data Science and AI" course. The project implements **Naive Bayes Classification for Email Spam Detection** with a focus on mathematical foundations and algorithm comparison.

## Project Structure

### Core Files
- `email_dataset.csv` - Original email server log dataset (5,000 records)
- `generate_synthetic_email_data.py` - Synthetic data generator maintaining statistical distributions
- `synthetic_email_dataset.csv` - Generated synthetic dataset for safe academic use

### Dataset Schema
Both datasets contain 12 fields:
- **Target Variable**: `Spam Detection` (empty = legitimate, "Moderate" = spam)
- **Key Features**: `Spam Score` (0-156), `Status` (email processing state), `Subject`, `From (Header)`
- **Supporting Fields**: `From (Envelope)`, `To`, `Sent Date/Time`, `IP Address`, `Attachment`, `Route`, `Info`

### Statistical Properties
- **Total Records**: 5,000 emails
- **Class Balance**: ~50% legitimate, ~50% spam (ideal for Naive Bayes)
- **Feature Types**: Numerical (`Spam Score`), categorical (`Status`), text (`Subject`, `From`)

## Development Commands

### Data Generation and Analysis
```bash
# Generate new synthetic dataset (overwrites existing synthetic_email_dataset.csv)
python3 generate_synthetic_email_data.py

# Quick dataset analysis
python3 -c "
import csv
from collections import Counter
with open('synthetic_email_dataset.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    print(f'Total: {len(rows)}')
    spam_count = sum(1 for r in rows if r['Spam Detection'] == 'Moderate')
    print(f'Spam: {spam_count}, Legitimate: {len(rows)-spam_count}')
    status_dist = Counter(r['Status'] for r in rows)
    print('Status distribution:', dict(status_dist.most_common(3)))
"

# Test synthetic data generator without overwriting files
python3 -c "
from generate_synthetic_email_data import SyntheticEmailGenerator
gen = SyntheticEmailGenerator(seed=123)
sample_data = gen.generate_dataset(num_records=100)
gen.print_statistics(sample_data)
"
```

### Environment Setup
```bash
# Virtual environment (optional - no external dependencies needed)
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Verify Python version compatibility
python3 --version  # Requires Python 3.7+
```

## Assignment Requirements

### Academic Constraints
- **Libraries**: Python built-in libraries only (no numpy, pandas, sklearn for implementation)
- **Allowed Tools**: Weka, Orange, scikit-learn for comparison/validation only
- **Implementation**: Hand-coded Naive Bayes classifier required

### Expected Deliverables
1. **Hand-coded Naive Bayes Classifier**
   - Manual probability calculations P(feature|class), P(class)
   - Bayes' theorem implementation from scratch
   - Feature extraction from text fields

2. **Sklearn Comparison**
   - Direct comparison with sklearn.naive_bayes
   - Performance metrics analysis
   - Algorithm validation

3. **Mathematical Analysis**
   - Feature independence assumption discussion
   - Conditional probability analysis
   - Optimization techniques exploration

## Code Architecture Notes

### Synthetic Data Generator Design
- **SyntheticEmailGenerator Class**: Main orchestrator with statistical preservation
- **Distribution Preservation**: Maintains original dataset proportions using probability sampling
- **Feature Correlation**: Realistic relationships (e.g., high spam scores → spam classification)
- **Reproducibility**: Fixed seed (42) for consistent academic results

### Feature Engineering Opportunities
```python
# Text features from Subject field
- Word count, capitalization patterns
- Spam keywords frequency
- Punctuation analysis

# Email features from From fields
- Domain reputation patterns
- Email address structure analysis
- Sender authenticity indicators

# Numerical features
- Spam Score thresholds
- IP address geolocation patterns
- Temporal patterns from datetime
```

### Naive Bayes Implementation Strategy
1. **Data Preprocessing**: Clean text, extract features, handle missing values
2. **Probability Calculation**:
   ```python
   P(spam|features) = P(features|spam) * P(spam) / P(features)
   ```
3. **Laplace Smoothing**: Handle zero probabilities for unseen feature combinations
4. **Classification**: Apply Bayes' theorem for prediction
5. **Validation**: Cross-validation, confusion matrix, accuracy metrics

## Mathematical Focus Areas

### Probability Theory Application
- **Conditional Probability**: P(feature|class) calculations
- **Bayes' Theorem**: Posterior probability computation
- **Prior Probabilities**: Class distribution analysis

### Independence Assumption Analysis
- **Feature Correlation**: Analyze violations of independence assumption
- **Impact Assessment**: How correlations affect classification accuracy
- **Mitigation Strategies**: Techniques to handle dependent features

### Optimization Concepts
- **Maximum Likelihood Estimation**: Parameter optimization
- **Log Probabilities**: Numerical stability in calculations
- **Cross-Validation**: Model selection and hyperparameter tuning

## Testing Strategy

### Validation Approach
```bash
# Split synthetic dataset for training/testing
# Implement k-fold cross-validation manually
# Compare hand-coded vs sklearn results
# Analyze feature importance and independence violations
```

### Performance Metrics
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix analysis
- ROC curves (if implementing probability scores)
- Feature importance ranking

## Academic Documentation Standards

### Code Requirements
- Comprehensive comments explaining mathematical concepts
- Clean, readable code structure suitable for academic evaluation
- Test cases on small examples as specified
- Mathematical derivations in comments where applicable

### Analysis Requirements
- In-depth independence assumption discussion
- Detailed comparison methodology
- Mathematical justification for design decisions
- Performance analysis with statistical significance

## Implementation Guidance

### Key Architecture Points
- **Single File Implementation**: The `SyntheticEmailGenerator` class is self-contained with all functionality
- **Statistical Preservation**: Uses probability sampling to maintain original dataset distributions
- **Academic Constraints**: Deliberately uses only Python standard library (csv, random, datetime, re, typing)
- **Reproducible Results**: Fixed seed (42) ensures consistent output for academic evaluation

### When Working with This Codebase
- **Modifying Data Generation**: Edit the class methods in `generate_synthetic_email_data.py` for different distributions
- **Testing Changes**: Use the test command above to verify modifications without overwriting datasets
- **Adding Features**: Extend the `generate_record()` method for additional email attributes
- **Dataset Analysis**: Both original and synthetic datasets follow identical 12-field schema

### Common Tasks
```bash
# Compare original vs synthetic distributions
python3 -c "
import csv
orig = list(csv.DictReader(open('email_dataset.csv')))
synth = list(csv.DictReader(open('synthetic_email_dataset.csv')))
print(f'Original spam: {sum(1 for r in orig if r[\"Spam Detection\"] == \"Moderate\")}')
print(f'Synthetic spam: {sum(1 for r in synth if r[\"Spam Detection\"] == \"Moderate\")}')
"

# Extract features for Naive Bayes implementation
python3 -c "
import csv
data = list(csv.DictReader(open('synthetic_email_dataset.csv')))
subjects = [r['Subject'] for r in data if r['Spam Detection'] == 'Moderate'][:5]
print('Sample spam subjects:', subjects)
"
```