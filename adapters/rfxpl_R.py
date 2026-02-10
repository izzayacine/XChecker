"""
RFxpl explainer implementation as explainer R.

This module provides a wrapper around RFxpl to implement the BaseExplainerR interface.
"""

from typing import Set, Tuple, Optional, Any
import numpy as np
import sys
import os
import subprocess
import tempfile


from ..checker.explainer import ReferenceExplainer
from ..checker.model import MLModel
from ..checker.types import Instance

# Add RFxpl to path
#sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../RFxpl'))

from pysat.solvers import Solver
from xrf import XRF
from xrf.explain import SATExplainer


class RF_Model(MLModel):

    def __init__(self, cls) -> None:
        super().__init__()
        self.model = cls

    def predict(self, instance: np.ndarray) -> int:
        # self.model.predict(instance.reshape((1, -1)))[0]
        return self.model.predict(instance)

    def nfeatures(self) -> int:
        return len(self.model.feature_names)

    def nclasses(self) -> int:
        return len(self.model.targets)
        

class RFxplExplainerR(ReferenceExplainer):
    """
    RFxpl explainer as reference explainer R.
    
    Implements all required queries for explainer R:
    - findaxp, findcxp, iswaxp, iswcxp, prooftrace
    """
    
    def __init__(self, model, feature_names, target_names, 
                 domains=None, verbose=False):
        """
        Initialize RFxpl explainer.
        
        Args:
            rf_model: RFxpl Random Forest model (RFSklearn instance)
            feature_names: List of feature names
            target_names: List of target/class names
            domains: Dictionary with min/max values for features
            verbose: Enable verbose output
        """
        super().__init__(model, domains)
        self.feature_names = feature_names
        self.target_names = target_names
        #self.feature_domains = domains
        self.verbose = verbose
        
        # Initialize XRF explainer
        self.xrf = XRF(model, feature_names, target_names)
        self.xrf.verbose = verbose
        
        # Note: We create explainers per-instance to avoid state issues
    
    def _get_explainer(self, instance: np.ndarray):
        """Get or create SATExplainer for the current instance."""
        # Always create a new explainer to avoid state issues
        # Encode the instance
        self.xrf.encode(instance, 'sat')
        
        inpvals = self.xrf.readable_data(instance)
        preamble = []
        for f, v in zip(self.xrf.feature_names, inpvals):
            if f not in str(v):
                preamble.append('{0} = {1}'.format(f, v))
            else:
                preamble.append(v)
        
        inps = self.xrf.ffnames
        explainer = SATExplainer(
            self.xrf.enc, inps, preamble, self.target_names, 
            verb=1 if self.verbose else 0
        )
        explainer.prepare_selectors(instance)
        
        return explainer
    
    def findaxp(self, instance: Instance) -> Set[int]:
        """Find an AXp using RFxpl."""
        if isinstance(instance, tuple):
            instance, _ = instance
        expl = self.xrf.explain(instance, xtype='abd', etype='sat', smallest=False)
        return set(expl)
    
    def findcxp(self, instance: Instance) -> Set[int]:
        """Find a CXp using RFxpl."""
        if isinstance(instance, tuple):
            instance, _ = instance
        expl = self.xrf.explain(instance, xtype='con', etype='sat', smallest=False)
        return set(expl)
    
    def iswaxp(self, instance: Instance, 
              expl: Set[int]) -> Tuple[bool, Optional[np.ndarray], Optional[Any]]:
        """
        Check if expl form a WAXp.
        
        Returns:
            (is_waxp, witness, proof_trace)
        """
        if isinstance(instance, tuple):
            instance, pred = instance

        explainer = self._get_explainer(instance)
        
        # Convert expl to list format expected by RFxpl
        # RFxpl uses 0-indexed features, but explanations might be 1-indexed
        expl_list = sorted(list(expl))
        
        # Prepare assumptions for the features
        assums = explainer.assums[:]
        for i, slt in enumerate(assums):
            feat = explainer.sel2fid[slt]
            if feat not in expl_list:
                assums[i] = -assums[i]
        
        # Check if UNSAT (WAXp) or SAT (not WAXp)
        explainer.slv = Solver(name="glucose3", with_proof=True)
        explainer.slv.append_formula(explainer.cnf)
        
        is_sat = explainer.slv.solve(assumptions=assums)
        
        witness, proof_trace = None, None
        if is_sat:
            # SAT - not a WAXp, return witness
            model = explainer.slv.get_model()
            witness = explainer.generate_wit(instance, model, self.feature_domains)
            explainer.slv.delete()
            explainer.slv = None
            #return False, witness, None
        else:
            # UNSAT - is a WAXp, return proof trace
            proof_trace = explainer.slv.get_proof()
            #==========================
            cnf = explainer.cnf.copy()
            cnf.extend([[lit] for lit in assums])
            assert proof_trace
            cnf.to_file("/tmp/encoding.cnf")   
            #=========================         
            explainer.slv.delete()
            explainer.slv = None
            #return True, None, proof_trace
        
        return (not is_sat), witness, proof_trace

    def iswcxp(self, instance: Instance,
              features: Set[int]) -> Tuple[bool, Optional[np.ndarray], Optional[Any]]:
        """
        Check if features form a WCXp.
        
        Returns:
            (is_wcxp, witness, proof_trace)
        """
        if isinstance(instance, tuple):
            instance, pred = instance

        explainer = self._get_explainer(instance)
        
        # Convert features to list format
        expl_list = sorted(list(features))
        
        # For CXp, we negate the assumptions (free the features)
        assums = explainer.assums[:]
        for i, slt in enumerate(assums):
            feat = explainer.sel2fid[slt]
            if feat in expl_list:
                assums[i] = -assums[i]  # Free this feature
        
        # Check if SAT (WCXp) or UNSAT (not WCXp)
        explainer.slv = Solver(name="glucose3", with_proof=True)
        explainer.slv.append_formula(explainer.cnf)
        
        is_sat = explainer.slv.solve(assumptions=assums)
        
        witness, proof_trace = None, None
        if is_sat:
            # SAT - is a WCXp, return witness
            model = explainer.slv.get_model()
            witness = explainer.generate_wit(instance, model, self.feature_domains)
            explainer.slv.delete()
            explainer.slv = None
            return True, witness, None
        else:
            # UNSAT - not a WCXp, return proof trace
            proof_trace = explainer.slv.get_proof()
            explainer.slv.delete()
            explainer.slv = None
            return False, None, proof_trace

        return is_sat, witness, proof_trace

    def prooftrace(self, instance: Instance,
                  features: Set[int]) -> Optional[Any]:
        """Get proof trace for WAXp."""
        # if isinstance(instance, tuple):
        #     instance, pred = instance

        is_waxp, _, proof_trace = self.iswaxp(instance, features)
        if is_waxp:
            return proof_trace
        return None

    def witness(self, 
                expl: Set[int], 
                target_classes: Optional[Set[int]] = None) -> Optional[np.ndarray]:
        """
        Generate a witness point in feature space.
        
        Uses the most recent instance passed to iswaxp/iswcxp/prooftrace.
        Returns None if no instance is available.
        """
        if isinstance(instance, tuple):
            instance, pred = instance
        else:
            pred = self.model.predict(instance)
            #instance = (instance, pred)

        explainer = self._get_explainer(instance)
        expl_list = sorted(list(features))
        # we negate the assumptions (free the features)
        assums = explainer.assums[:]
        for i, slt in enumerate(assums):
            feat = explainer.sel2fid[slt]
            if feat in expl_list:
                assums[i] = -assums[i]  # Free this feature
        
        witness = None
        # Check if SAT
        with Solver(name="glucose3", bootstrap_with=explainer.cnf) as slv:
            if slv.solve(assumptions=assums):
                model = slv.get_model()
                witness = explainer.generate_wit(instance, model, self.feature_domains)

        return witness
    
    def validate_proof_with_drat(self, cnf, assumptions, proof_trace) -> Optional[bool]:
        """
        Validate a proof trace using DRAT-trim.
        
        This is a helper method that uses RFxpl's proof validation infrastructure.
        
        Args:
            cnf: CNF formula
            assumptions: List of assumption literals
            proof_trace: Proof trace (list of strings)
            
        Returns:
            True if verified, False if not verified, None if cannot verify
        """
        
        try:
            # Create temporary files
            with tempfile.NamedTemporaryFile(mode='w', suffix='.cnf', delete=False) as cnf_file:
                cnf_path = cnf_file.name
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.drat', delete=False) as drat_file:
                drat_path = drat_file.name
            
            # Write CNF + assumptions
            cnf_copy = cnf.copy()
            cnf_copy.extend([[lit] for lit in assumptions])
            cnf_copy.to_file(cnf_path)
            
            # Write proof trace
            with open(drat_path, 'w') as f:
                for line in proof_trace:
                    f.write(str(line) + "\n")
            
            # Find drat-trim
            drat_trim_path = None
            current_dir = os.path.dirname(os.path.abspath(__file__))        # adapters/
            possible_paths = [
                os.path.join(current_dir, "./drat-trim"),
                os.path.join(current_dir, "../checker/drat-trim"),
                os.path.join(current_dir, "../../RFxpl/drat-trim")
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    drat_trim_path = path
                    break
            
            if drat_trim_path is None:
                if self.verbose:
                    print("drat-trim not found, cannot validate proof")
                return None
            
            # Run drat-trim
            result = subprocess.run(
                [drat_trim_path, cnf_path, drat_path, '-f'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            output = result.stdout + result.stderr
            
            # Cleanup
            try:
                os.unlink(cnf_path)
                os.unlink(drat_path)
            except:
                pass
            
            if "s VERIFIED" in output:
                return True
            elif "s NOT VERIFIED" in output:
                return False
            else:
                if self.verbose:
                    print(f"drat-trim returned unexpected output: {output}")
                return None
                
        except Exception as e:
            if self.verbose:
                print(f"Proof validation error: {e}")
            return None
