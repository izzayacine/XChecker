"""
Example script for validating formal explainers.

This script demonstrates how to use the validation framework to validate
explanations from explainer T (e.g., PyXAI) using explainer R (RFxpl)
and explainer S (XReason).
"""

import numpy as np
import sys
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

# Add paths
sys.path.insert(0, os.path.join(PROJECT_ROOT, "RFxpl"))

from XChecker.checker import Validator, ValidationResult 
from XChecker.checker import FeatureDomains, NumericDomain, CategoricalDomain
from XChecker.adapters import RFxplExplainerR, XReasonExplainerS, PyXAIExplainerT
from XChecker.adapters.rfxpl_R import RF_Model

# Import RFxpl components
from xrf import RFSklearn, Dataset, Forest
from options import Options as RFOptions

sys.path.insert(0, os.path.join(PROJECT_ROOT, "XReason-RFs", "src"))
# Import XReason components (if available)
try:
    from xgbooster import XGBooster
    #from data import Data
    from options import Options as XReasonOptions
    XREASON_AVAILABLE = True
except ImportError:
    XREASON_AVAILABLE = False
    print("Warning: XReason not available")



def validate_explanation_example():
    """
    Example: Validate an AXp explanation from PyXAI using RFxpl and XReason.
    """
    
    # 1. Load the ML model
    # Example: Load RFxpl model
    #pickle_path = '../RFxpl/Classifiers/29_Pima/29_Pima_nbestim_3_maxdepth_4.mod.pkl'
    pickle_path = './experiments/tests/29_Pima/29_Pima_nbestim_5_maxdepth_4.mod.pkl'
    enc_path = './tests/29_Pima/29_Pima_nbestim_5_maxdepth_4.smt' 
    enc_path = os.path.join(PROJECT_ROOT, "XChecker/experiments/", enc_path)
    csv_path = './experiments/tests/29_Pima/29_Pima.csv'    
    
    if not os.path.exists(pickle_path):
        print(f"Model file not found: {pickle_path}")
        print("Please update the path to a valid model file")
        return

    if not os.path.exists(enc_path):
        print(f"SMT-Encoding file not found: {enc_path}")
        print("Please update the path to a valid file")
        return

    # laod dataset .csv
    if not os.path.exists(csv_path):
        print(f"Model file not found: {csv_path}")
        print("Please update the path to a valid dataset file")
        return    

    opts = RFOptions(None)
    opts.files = [pickle_path, csv_path]
    #rf_md = RFSklearn(from_file=opts.files[0])
    rf_md = RFSklearn(from_file=opts.files[0])
    #feature_names = rf_md.feature_names
    #target_names = rf_md.targets
                     
    data = Dataset(filename=opts.files[1], use_categorical=False)
    feature_names, target_names = data.features, data.targets
    forest = Forest(rf_md.estimators(), data.m_features_)
    # Get feature domain bounds if available (for witness generation)
    # data.X.shape = (rows, m)
    min_max_dom = [NumericDomain(data.X[:, i].min(), data.X[:, i].max()) 
                for i in range(data.X.shape[1])]
               
    
    # Create ML model wrapper
    ml_model = RF_Model(forest)
    
    # 2. Initialize explainer R (RFxpl)
    explainer_r = RFxplExplainerR(
        rf_md, feature_names, target_names,
        domains=min_max_dom,
        verbose=False
    )
    

    # 3. Initialize explainer S (XReason) - optional
    explainer_s = None
    if XREASON_AVAILABLE:
        try:
            #enc_path = None
            explainer_s = XReasonExplainerS(ml_model, rf_md.feature_names, target_names, encoding=enc_path)
        except Exception as e:
            print(f"Could not initialize XReason: {e}")
    
    # 4. Initialize explainer T (PyXAI) - template
    explainer_t = PyXAIExplainerT(
        model=ml_model,
        domains=min_max_dom,
        verbose=False
    )
    
    # 5. Create validator
    validator = Validator(
        explainer_t=explainer_t,
        explainer_r=explainer_r,
        explainer_s=explainer_s,
        ml_model=ml_model,
        verbose=True
    )
    
    # 6. Example instance and explanation to validate
    # This would typically come from PyXAI output
    instance = np.array([6.0, 148.0, 72.0, 35.0, 0.0, 33.6, 0.627, 50.0])  # Example

    # if opts.cat_data:
    #     explainer_r.xrf.ffnames = data.m_features_
    #     explainer_r.xrf.readable_sample = lambda x: data.readable_sample(data.transform_inverse(x)[0])
    #     instance = data.transform(np.array(instance))[0]    
    prediction = ml_model.predict(instance)
    
    # Example explanation from PyXAI (this would come from explainer_t.findaxp)
    # For demonstration, we'll use a dummy explanation
    expl = {0, 1, 2, 3, 4, 5, 6, 7}  # Example feature indices
    #expl = {1, 6, 7}
    
    print(f"Instance: {instance}")
    print(f"Prediction: {prediction} ({target_names[prediction]})")
    print(f"Explanation from T: {expl}")
    print("\n" + "="*60)
    print("Validating AXp explanation...")
    print("="*60 + "\n")
    
    # 7. Validate the explanation
    report = validator.validate_axp((instance, prediction), expl)
    
    # 8. Print results
    print("Validation Report:")
    print("-" * 60)
    print(report)
    print("-" * 60)
    
    if report.result == ValidationResult.CORRECT:
        print("✓ Explanation is VALID and MINIMAL")
    elif report.result == ValidationResult.NON_MINIMAL:
        print(f"⚠ Explanation is VALID but NOT MINIMAL")
        print(f"  Real explanation: {sorted(report.real_explanation)}")
        print(f"  Redundant features: {sorted(expl - report.real_explanation)}")
    elif report.result == ValidationResult.INCORRECT:
        print("✗ Explanation is INVALID")
        if report.witness is not None:
            print(f"  Witness: {report.witness}")
            print(f"  Witness prediction: {ml_model.predict(report.witness)}")
    else:
        print("✗ Validation ERROR")
        for error in report.errors:
            print(f"  Error: {error}")
    
    return report


def validate_cxp_example():
    """
    Example: Validate a CXp explanation.
    """
    # Similar to validate_explanation_example but using validate_cxp
    # This is left as an exercise for the user
    pass


if __name__ == '__main__':
    print("="*60)
    print("Formal Explainer Validation Framework - Example")
    print("="*60)
    print()
    
    # Run validation example
    try:
        report = validate_explanation_example()
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()
