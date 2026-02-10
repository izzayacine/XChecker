# Usage Guide for Validation Framework

## Quick Start

### 1. Basic Validation

```python
from XChecker.checker import Validator
from XChecker.adapters import RFxplExplainerR
from XChecker.adapters.rfxpl_R import RF_Model
from xrf import RFSklearn

# Load model
rf_md = RFSklearn(from_file='model.pkl')
ml_model = RF_Model(rf_md)

# Initialize explainer R
explainer_r = RFxplExplainerR(
    rf_md, feature_names, target_names,
    domains=min_max_dict
)

# Create validator
validator = Validator(
    explainer_t=None,  # Not needed if validating directly
    explainer_r=explainer_r,
    explainer_s=None,  # Optional
    ml_model=ml_model
)

# Validate explanation
instance = np.array([...])
prediction = ml_model.predict(instance)
expl = {0, 1, 2}  # Set of feature indices
report = validator.validate_axp((instance, prediction), expl)
```

Note: The validator expects the instance as a tuple `(np.ndarray, prediction)`.

### 2. With Explainer S (XReason)

```python
from XChecker.adapters import XReasonExplainerS

# Initialize XReason
explainer_s = XReasonExplainerS(
    model=ml_model,
    feature_names=feature_names,
    target_names=target_names,
    options=xgb_options
)

# Add to validator
validator = Validator(
    explainer_t=None,
    explainer_r=explainer_r,
    explainer_s=explainer_s,  # Now included
    ml_model=ml_model
)
```

### 3. Using Feature Domains for Witness Generation

Feature domains are required when generating witness points. They define the valid ranges/values for each feature.

```python
from XChecker.checker import NumericDomain, CategoricalDomain, FeatureDomains

# Create feature domains as a list (indexed by feature position)
feature_domains: FeatureDomains = [
    NumericDomain(min_=0.0, max_=100.0),      # feature 0
    NumericDomain(min_=-10.0, max_=10.0),   # feature 1
    CategoricalDomain(values=[0, 1, 2]),     # feature 2 (or ['A', 'B', 'C'] for strings)
    # ... more features
]

# Access domains directly by index
numeric_domain = feature_domains[0]  # NumericDomain
if isinstance(numeric_domain, NumericDomain):
    print(f"Range: [{numeric_domain.min_}, {numeric_domain.max_}]")

categorical_domain = feature_domains[2]  # CategoricalDomain
if isinstance(categorical_domain, CategoricalDomain):
    print(f"Values: {categorical_domain.values}")

# Use with witness function
witness = explainer_t.witness(
    features={0, 1, 2},
    feature_domains=feature_domains,
    target_classes={1, 2}  # Optional
)
```

### 3. Validating from Log Files

See `integration_example.py` for a complete example of validating explanations
from log files.

## Validation Results

The `ValidationReport` object contains:

- **result**: `ValidationResult` enum
  - `CORRECT`: Explanation is valid and minimal
  - `INCORRECT`: Explanation is invalid
  - `NON_MINIMAL`: Explanation is valid but not minimal
  - `ERROR`: Validation error occurred

- **real_explanation**: The actual minimal explanation found
- **witness**: Witness point if found (for invalid explanations)
- **proof_verified**: Whether proof was verified (True/False/None)
- **num_witness_checks**: Number of witness validations performed
- **num_proof_checks**: Number of proof verifications performed
- **errors**: List of error messages

## Adapting for Your Use Case

### Using PyXAI as Explainer T

1. Implement the `PyXAIExplainerT` class based on your PyXAI version
2. Implement `findaxp`, `findcxp`, and `witness` methods
3. Use the validator to validate PyXAI's explanations

### Adding Other Explainers

1. Create a new class inheriting from the appropriate base class:
   - `TargetExplainer` for untrusted explainers
   - `ReferenceExplainer` for reference explainers
   - `SecondaryExplainer` for independent explainers

2. Implement all required methods

3. Use with the validator

## Integration with Existing Scripts

The framework is designed to work alongside existing validation scripts.
You can:

1. Use `integration_example.py` as a template
2. Replace direct validation calls with validator calls
3. Keep the same output format for compatibility

## Troubleshooting

### DRAT-trim not found

The validator will look for `drat-trim` in common locations:
- `XChecker/drat-trim`
- `XChecker/adapters/drat-trim`
- `RFxpl/drat-trim`

If not found, proof validation will return `None` (unknown).

### XReason not available

If XReason is not available, you can still use the validator without
explainer S. The validator will skip S-based checks.

### Memory issues

If you encounter memory issues with large models:
- Process explanations in batches
- Clear explainer state between validations
- Use smaller models for testing
