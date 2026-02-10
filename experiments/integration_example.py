"""
Integration example showing how to use the validator with existing check scripts.

This example shows how to adapt existing validation scripts (like check_axp.py)
to use the new validator framework.
"""

import numpy as np
import pandas as pd
import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Add paths
sys.path.insert(0, os.path.join(PROJECT_ROOT, "RFxpl"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "XReason-RFs", "src"))

from XChecker.checker import Validator, ValidationResult
# from validator.explainers import RFxplExplainerR
# from validator.explainers.rfxpl_explainer import RFxplMLModel
from XChecker.adapters import RFxplExplainerR
from XChecker.adapters.rfxpl_R import RF_Model
from XChecker.checker import FeatureDomains, NumericDomain, CategoricalDomain

from xrf import RFSklearn, Forest
from options import Options


def validate_from_log_file(log_path, model_path, dataset_path, output_path):
    """
    Validate explanations from a log file (similar to check_axp.py).
    
    Args:
        log_path: Path to log file with explanations
        model_path: Path to model pickle file
        dataset_path: Path to dataset CSV (for min/max values)
        output_path: Path to output validation results
    """
    
    # 1. Load model
    opts = Options(None)
    opts.files = model_path
    rf_md = RFSklearn(from_file=opts.files)
    feature_names = rf_md.feature_names
    target_names = rf_md.targets
    
    # 2. Load min/max values from dataset
    df = pd.read_csv(dataset_path)
    df_X = df.iloc[:, :-1]
    feature_domains: FeatureDomains = []
    for col in df_X.columns:
        series = df_X[col]
        if pd.api.types.is_numeric_dtype(series):
            domain = NumericDomain(
                min_=float(series.min()),
                max_=float(series.max())
            )
        else:
            domain = CategoricalDomain(
                values=series.dropna().unique()
            )
        feature_domains.append(domain)
    
    # 3. Initialize explainers
    forest = Forest(rf_md.estimators(), feature_names)
    ml_model = RF_Model(forest)
    explainer_r = RFxplExplainerR(
        rf_md, feature_names, target_names,
        domains=feature_domains,
        verbose=False
    )
    
    # 4. Create validator (without explainer S for now)
    validator = Validator(
        explainer_t=None,  # We'll validate explanations directly
        explainer_r=explainer_r,
        explainer_s=None,
        ml_model=ml_model,
        verbose=False
    )
    
    # 5. Parse log file
    i_list = []
    i_pred_list = []
    expl_list = []
    
    try:
        with open(log_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("i:"):
                    values = list(map(float, line[2:].strip().split(',')))
                    i_list.append(values)
                elif line.startswith("pred:"):
                    values = int(line[5:].strip())
                    i_pred_list.append(values)
                elif line.startswith("expl:"):
                    raw = line[5:].strip().strip('[]')
                    if raw.lower() == "timeout":
                        expl_list.append("timeout")
                    else:
                        values = list(map(int, raw.split(','))) if raw else []
                        expl_list.append(values)
    except FileNotFoundError:
        print(f"File not found: {log_path}")
        return
    
    # 6. Validate each explanation
    tested = set()
    invalid_xp = 0
    valid_xp = 0
    num_wxp = 0
    num_to = 0
    output_str = ''
    
    for i, (inst, expl) in enumerate(zip(i_list, expl_list)):
        inst_tuple = tuple(inst)
        if inst_tuple in tested:
            continue
        tested.add(inst_tuple)
        
        if expl == "timeout":
            num_to += 1
            continue
        
        if not expl:
            invalid_xp += 1
            continue
        
        # Convert to 0-indexed if needed (PyXAI uses 1-indexed)
        expl_set = {idx - 1 for idx in expl}
        instance = np.array(inst)
        prediction = i_pred_list[i] if i < len(i_pred_list) else ml_model.predict(instance)
        
        # Validate using the framework
        report = validator.validate_axp((instance, prediction), expl_set)
        
        # Update statistics
        if report.result == ValidationResult.CORRECT:
            valid_xp += 1
        elif report.result == ValidationResult.INCORRECT:
            invalid_xp += 1
        elif report.result == ValidationResult.NON_MINIMAL:
            num_wxp += 1
        
        # Write output
        output_str += f"i: {','.join(f'{x}' for x in inst)}\n"
        output_str += f"PyXAI: {expl}\n"
        output_str += f"cert: {report.proof_verified if report.proof_verified is not None else 'N/A'}\n"
        output_str += f"Consistent ML referees: {prediction == ml_model.predict(instance)}\n"
        
        if report.result == ValidationResult.INCORRECT:
            if report.witness is not None:
                output_str += f"RFxpl sub-min wit invalid: {','.join(f'{x}' for x in report.witness)}\n"
            else:
                output_str += f"DRAT-TRIM proof failed\n"
        else:
            if expl != "timeout":
                output_str += f"AXp: {sorted(report.real_explanation)}\n"
                if report.witness is not None:
                    output_str += f"RFxpl wit: {','.join(f'{x}' for x in report.witness)}\n"
                    if ml_model.predict(report.witness) == ml_model.predict(instance):
                        output_str += f"RFxpl wit invalid!\n"
        
        output_str += f"nof. wits: {report.num_witness_checks}\n"
        output_str += f"nof. proofs: {report.num_proof_checks}\n"
        if expl != "timeout":
            output_str += f"nof. waxp calls: {report.num_witness_checks + report.num_proof_checks}\n"
            output_str += f"nof. buggy: {len(expl_set) - len(report.real_explanation)}\n"
            if len(expl_set) > 0:
                output_str += f"Redundant %: {(len(expl_set) - len(report.real_explanation)) * 100 / len(expl_set):.1f}\n\n"
            else:
                output_str += f"Redundant %: NA\n\n"
        else:
            output_str += f"Redundant %: NA\n\n"
    
    # Write summary
    output_str += f"nof. valid: {valid_xp}\n"
    output_str += f"nof. invalid: {invalid_xp}\n"
    output_str += f"nof. non-minimal: {num_wxp}\n"
    output_str += f"nof. timeout: {num_to}\n"
    
    # Write to output file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(output_str)
    
    print(f"Validation complete. Results written to {output_path}")
    print(f"Valid: {valid_xp}, Invalid: {invalid_xp}, Non-minimal: {num_wxp}, Timeout: {num_to}")


if __name__ == '__main__':
    # Example usage
    if len(sys.argv) < 5:
        print("Usage: python integration_example.py <log_path> <model_path> <dataset_path> <output_path>")
        sys.exit(1)
    
    log_path = sys.argv[1]
    model_path = sys.argv[2]
    dataset_path = sys.argv[3]
    output_path = sys.argv[4]
    
    validate_from_log_file(log_path, model_path, dataset_path, output_path)
