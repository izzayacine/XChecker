"""
Main validator class implementing the validation framework.

This module implements the validation approach described in the paper,
validating explanations from explainer T using explainer R, and validating
explainer R's results using proof traces and explainer S.
"""

import subprocess
import tempfile
import os

from typing import Set, Optional, Tuple, Dict, Any, List
import numpy as np
from enum import Enum

from .explainer import TargetExplainer, ReferenceExplainer, SecondaryExplainer
from .model import MLModel
from .types import Instance

class ValidationResult(Enum):
    """Result of validation."""
    CORRECT = "correct"
    INCORRECT = "incorrect"
    NON_MINIMAL = "non_minimal"
    ERROR = "error"


class ValidationReport:
    """Report from validation process."""
    
    def __init__(self):
        self.result: Optional[ValidationResult] = None
        self.is_valid: bool = False
        self.is_minimal: bool = False
        self.real_explanation: Set[int] = set()
        self.witness: Optional[np.ndarray] = None
        self.proof_verified: Optional[bool] = None
        self.s_agreement: Optional[bool] = None
        self.errors: List[str] = []
        self.num_witness_checks: int = 0
        self.num_proof_checks: int = 0
        
    def __str__(self):
        lines = [
            f"Result: {self.result.value if self.result else 'UNKNOWN'}",
            f"Valid: {self.is_valid}",
            f"Minimal: {self.is_minimal}",
            f"Real explanation: {sorted(self.real_explanation)}",
            f"Witness checks: {self.num_witness_checks}",
            f"Proof checks: {self.num_proof_checks}",
        ]
        if self.errors:
            lines.append(f"Errors: {self.errors}")
        return "\n".join(lines)


class Validator:
    """
    Validator for formal explainers.
    
    Implements the validation approach from the paper:
    - Validates explanations from explainer T using explainer R
    - Validates explainer R's results using proof traces and explainer S
    """
    
    def __init__(self, 
                 ml_model: MLModel,
                 explainer_t: Optional[TargetExplainer],
                 explainer_r: ReferenceExplainer,
                 explainer_s: Optional[SecondaryExplainer] = None,
                 verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            ml_model: ML model M for witness validation
            explainer_t: Untrusted explainer T
            explainer_r: Reference explainer R
            explainer_s: Second explainer S (optional)
            verbose: Enable verbose output
        """
        self.explainer_t = explainer_t
        self.explainer_r = explainer_r
        self.explainer_s = explainer_s
        self.ml_model = ml_model
        self.verbose = verbose
    
    def validate_axp(self, 
                    instance: Instance,
                    expl: Set[int],
                    prediction: Optional[int] = None) -> ValidationReport:
        """
        Validate an abductive explanation (AXp) from explainer T.
        
        An AXp X is valid if:
        (i) X is a WAXp
        (ii) For any t in X, X\{t} is not a WAXp (minimality)
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            expl: Set of feature indices claimed to be an AXp
            prediction: Predicted class (if None, will query ML model)
            
        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport()
        
        # if self.ml_model is None:
        #     report.errors.append("ML model not provided")
        #     report.result = ValidationResult.ERROR
        #     return report
        
        if isinstance(instance, tuple):
            instance, instance_pred = instance
            if prediction is None:
                prediction = instance_pred

        if prediction is None:
            prediction = self.ml_model.predict(instance)

        instance_with_pred = (instance, prediction)
        
        # Step 1: Check if expl is a WAXp using explainer R
        is_waxp, witness, proof_trace = self.explainer_r.iswaxp(
            instance_with_pred, expl
        )
        
        if not is_waxp:
            # Case Ra2: Not a WAXp - validate witness
            if witness is not None:
                report.num_witness_checks += 1
                witness_pred = self.ml_model.predict(witness)
                if witness_pred == prediction:
                    report.errors.append(
                        "WAXp check failed: witness has same prediction as original"
                    )
                    report.result = ValidationResult.INCORRECT
                    report.witness = witness
                    return report
                else:
                    report.errors.append("Explanation is not a WAXp (witness confirmed)")
                    report.result = ValidationResult.INCORRECT
                    report.witness = witness
                    return report
            else:
                report.errors.append("WAXp check failed: no witness provided")
                report.result = ValidationResult.ERROR
                return report
        else:
            # Case Ra1: Is a WAXp - validate proof and check with S
            if proof_trace is not None:
                report.num_proof_checks += 1
                proof_verified = self._validate_proof(proof_trace, instance, expl)
                report.proof_verified = proof_verified
                
                if not proof_verified:
                    report.errors.append("WAXp proof verification failed")
                    report.result = ValidationResult.ERROR
                    return report
            
            # Check with explainer S if available
            if self.explainer_s is not None:
                s_is_waxp = self.explainer_s.iswaxp(instance_with_pred, expl)
                report.s_agreement = s_is_waxp
                
                if not s_is_waxp:
                    report.errors.append("WAXp check: explainer S disagrees")
                    report.result = ValidationResult.ERROR
                    return report
        
        # Step 2: Check minimality - for each feature, check if removing it
        # still gives a WAXp
        minimal_expl = set(expl)
        x_r_error = False
        
        for feature in list(expl):
            to_test = minimal_expl - {feature}
            is_reducible, witness, proof_trace = self.explainer_r.iswaxp(instance_with_pred, to_test)
            if is_reducible:
                # Removing this feature still gives WAXp - not minimal
                if proof_trace is not None:
                    report.num_proof_checks += 1
                    proof_verified = self._validate_proof(proof_trace, instance, to_test)
                    if not proof_verified:
                        report.errors.append(
                            f"Sub-minimality check: proof verification failed for feature {feature}"
                        )
                        x_r_error = True
                        #continue
                
                # Validate with S if available
                if self.explainer_s is not None:
                    s_is_waxp = self.explainer_s.iswaxp(instance_with_pred, to_test)
                    if not s_is_waxp:
                        report.errors.append(
                            f"Sub-minimality check: explainer S disagrees for feature {feature}"
                        )
                        x_r_error = True
                        #continue
                
                # Feature can be removed - not minimal
                minimal_expl.remove(feature)
            else:
                # Check witness if provided
                if witness is not None:
                    report.num_witness_checks += 1
                    witness_pred = self.ml_model.predict(witness)
                    if witness_pred == prediction:
                        report.errors.append(
                            f"Sub-minimality check: witness invalid for feature {feature}"
                        )
                        report.witness = witness
                        report.result = ValidationResult.ERROR
                        return report
        
        # Finalize report
        report.real_explanation = minimal_expl #if not x_r_error else set()
        report.is_valid = (len(minimal_expl) == len(expl))
        report.is_minimal = report.is_valid
        
        if report.is_valid:
            report.result = ValidationResult.CORRECT
        elif len(minimal_expl) > 0:
            report.result = ValidationResult.NON_MINIMAL
        else:
            report.result = ValidationResult.INCORRECT
        
        return report
    
    def validate_cxp(self,
                    instance: Instance,
                    expl: Set[int],
                    prediction: Optional[int] = None) -> ValidationReport:
        """
        Validate a contrastive explanation (CXp) from explainer T.
        
        A CXp Y is valid if:
        (i) Y is a WCXp
        (ii) For any t in Y, Y\{t} is not a WCXp (minimality)
        
        Args:
            instance: Tuple of (feature vector, predicted class label)
            expl: Set of feature indices claimed to be a CXp
            prediction: Predicted class (if None, will query ML model)
            
        Returns:
            ValidationReport with validation results
        """
        report = ValidationReport()
        
        # if self.ml_model is None:
        #     report.errors.append("ML model not provided")
        #     report.result = ValidationResult.ERROR
        #     return report
        
        if isinstance(instance, tuple):
            instance, instance_pred = instance
            if prediction is None:
                prediction = instance_pred

        if prediction is None:
            prediction = self.ml_model.predict(instance)

        instance_with_pred = (instance, prediction)
        
        # Step 1: Check if expl is a WCXp using explainer R
        is_wcxp, witness, proof_trace = self.explainer_r.iswcxp(
            instance_with_pred, expl
        )
        
        if not is_wcxp:
            # Case Rc2: Not a WCXp - validate proof
            if proof_trace is not None:
                report.num_proof_checks += 1
                proof_verified = self._validate_proof(proof_trace, instance, expl)
                report.proof_verified = proof_verified
                
                if not proof_verified:
                    report.errors.append("WCXp proof verification failed")
                    report.result = ValidationResult.ERROR
                    return report
            
            # Check with explainer S if available
            if self.explainer_s is not None:
                s_is_wcxp = self.explainer_s.iswcxp(instance_with_pred, expl)
                report.s_agreement = s_is_wcxp
                
                if not s_is_wcxp:
                    report.errors.append("WCXp check: explainer S disagrees")
                    report.result = ValidationResult.ERROR
                    return report
            
            report.errors.append("Explanation is not a WCXp")
            report.result = ValidationResult.INCORRECT
            return report
        else:
            # Case Rc1: Is a WCXp - validate witness
            if witness is not None:
                report.num_witness_checks += 1
                witness_pred = self.ml_model.predict(witness)
                if witness_pred == prediction:
                    report.errors.append(
                        "WCXp check failed: witness has same prediction as original"
                    )
                    report.result = ValidationResult.INCORRECT
                    report.witness = witness
                    return report
                else:
                    # Witness confirmed - different prediction
                    report.witness = witness
            else:
                report.errors.append("WCXp check: no witness provided")
                report.result = ValidationResult.ERROR
                return report
        
        # Step 2: Check minimality - for each feature, check if removing it
        # still gives a WCXp
        minimal_expl = set(expl)
        
        for feature in list(expl):
            to_test = minimal_expl - {feature}
            is_reducible, witness, proof_trace = \
                self.explainer_r.iswcxp(instance_with_pred, to_test)
            
            if is_reducible:
                # Check witness
                if witness is not None:
                    report.num_witness_checks += 1
                    witness_pred = self.ml_model.predict(witness)
                    if witness_pred == prediction:
                        report.errors.append(
                            f"Sub-minimality check: witness invalid for feature {feature}"
                        )
                        report.witness = witness
                        report.result = ValidationResult.ERROR
                        return report
                
                # Feature can be removed - not minimal
                minimal_expl.remove(feature)
            else:
                # Validate proof if provided
                if proof_trace is not None:
                    report.num_proof_checks += 1
                    proof_verified = self._validate_proof(
                        proof_trace, instance, to_test
                    )
                    if not proof_verified:
                        report.errors.append(
                            f"Sub-minimality check: proof verification failed for feature {feature}"
                        )
                        continue
                
                # Validate with S if available
                if self.explainer_s is not None:
                    s_is_wcxp = self.explainer_s.iswcxp(instance_with_pred, to_test)
                    if not s_is_wcxp:
                        report.errors.append(
                            f"Sub-minimality check: explainer S disagrees for feature {feature}"
                        )
                        continue
        
        # Finalize report
        report.real_explanation = minimal_expl
        report.is_valid = (len(minimal_expl) == len(expl))
        report.is_minimal = report.is_valid
        
        if report.is_valid:
            report.result = ValidationResult.CORRECT
        elif len(minimal_expl) > 0:
            report.result = ValidationResult.NON_MINIMAL
        else:
            report.result = ValidationResult.INCORRECT
        
        return report
    
    def _validate_proof(self, proof_trace: Any, instance: np.ndarray, 
                       features: Set[int]) -> bool:
        """
        Validate a proof trace using DRAT-trim or similar.
        
        This method can be overridden by concrete validators that know
        how to validate specific proof formats. The default implementation
        attempts to use DRAT-trim if the proof trace is in the expected format.
        
        Args:
            proof_trace: Proof trace from explainer R
            instance: Feature vector
            features: Set of feature indices
            
        Returns:
            True if proof is valid, False if invalid, None if cannot verify
        """
        # Default implementation - attempts DRAT validation if proof_trace
        # is a list of strings (DRAT format)
        if isinstance(proof_trace, list) and len(proof_trace) > 0:
            # Try to validate using DRAT-trim

            try:
                # Create temporary files
                with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as cnf_file:
                    cnf_path = cnf_file.name
                
                with tempfile.NamedTemporaryFile(mode='w', suffix='.drat', delete=False) as drat_file:
                    drat_path = drat_file.name
                
                # Write proof trace to DRAT file
                with open(drat_path, 'w') as f:
                    for line in proof_trace:
                        f.write(str(line) + "\n")
                
                # Note: We need the CNF and assumptions from explainer R
                # This is a simplified version - in practice, you'd get the CNF
                # from the explainer R's internal state
                # For now, we'll assume the proof trace includes necessary info
                cnf_path = "/tmp/encoding.cnf"
                
                # Try to run drat-trim (if available)
                drat_trim_path = "./drat-trim"  # Default path
                if not os.path.exists(drat_trim_path):
                    # Try common locations
                    current_dir = os.path.dirname(os.path.abspath(__file__))
                    possible_paths = [
                        os.path.join(current_dir, "./drat-trim"),
                        os.path.join(current_dir, "../adapters/drat-trim"),
                        os.path.join(current_dir, "../../RFxpl/drat-trim")
                    ]
                    for path in possible_paths:
                        if os.path.exists(path):
                            drat_trim_path = path
                            break
                
                if os.path.exists(drat_trim_path):
                    # Note: This is a simplified version
                    # In practice, you need the full CNF + assumptions
                    # This would typically be done by the explainer R wrapper
                    result = subprocess.run(
                        [drat_trim_path, cnf_path, drat_path, '-f'],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    output = result.stdout + result.stderr
                    if "s VERIFIED" in output:
                        return True
                    elif "s NOT VERIFIED" in output:
                        return False
                
                # Cleanup
                try:
                    os.unlink(cnf_path)
                    os.unlink(drat_path)
                except:
                    pass
                    
            except Exception as e:
                if self.verbose:
                    print(f"Proof validation error: {e}")
                return None
        
        # If we can't validate, return None (unknown)
        if self.verbose:
            print("Proof validation not implemented for this proof format")
        return None  # Unknown - let the caller decide
