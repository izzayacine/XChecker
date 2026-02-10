"""
XReason explainer implementation as explainer S (SMT-based).

This module provides a wrapper around XReason to implement the BaseExplainerS interface.
"""

from typing import Set
import numpy as np
import sys
import os

import re
from pathlib import Path

from ..checker.explainer import SecondaryExplainer
from ..checker.types import Instance

# Add XReason-RFs to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../XReason-RFs/src'))

from xgbooster import XGBooster, SMTValidator
from options import Options


class XReasonExplainerS(SecondaryExplainer):
    """
    XReason explainer as second explainer S (SMT-based).
    
    Implements required queries for explainer S:
    - iswaxp, iswcxp
    """
    
    def __init__(self, model, feature_names, target_names, 
                 options=None, encoding=None, verbose=False):
        """
        Initialize XReason explainer.
        
        Args:
            model: ML model instance (passed to base explainer)
            feature_names: List of feature names
            target_names: List of target/class names
            options: Options object for XGBooster (optional)
            encoding: Path to SMT encoding file (required if options is None)
            verbose: Enable verbose output
        """
        super().__init__(model)
        self.feature_names = feature_names
        self.target_names = target_names
        self.verbose = verbose
        
        opts = self._build_options(options, encoding)
        from_model, from_encoding = self._resolve_sources(opts)
        
        self.xgb = XGBooster(opts, from_model=from_model, from_encoding=from_encoding)

        self.xgb.feature_names = self.feature_names
        self.xgb.target_name = list(range(len(self.target_names)))

        # Initialize SMT-based checker 
        self.smt_checker = SMTValidator(
            self.xgb.enc, self.feature_names, len(self.target_names), self.xgb
        )
        

    def _build_options(self, options, encoding):
        if options is not None:
            return options

        if encoding is None:
            raise ValueError("encoding must be provided when options is None")

        enc_path = Path(encoding).name
        match = re.search(r"nbestim_(\d+)_maxdepth_(\d+)", enc_path)
        if not match:
            raise ValueError(
                "Could not extract nbestim and maxdepth from encoding filename"
            )

        opts = Options(None)
        opts.n_estimators = int(match.group(1))
        opts.maxdepth = int(match.group(2))
        opts.files = [encoding]
        opts.encode = encoding
        return opts

    def _resolve_sources(self, opts):
        encoding_flag = getattr(opts, "encode", None)
        if encoding_flag is None:
            raise AttributeError("options must define the 'encoding' attribute")

        if not getattr(opts, "files", None):
            raise ValueError("options.files must contain the model/encoding path")
        
        if encoding_flag in [None, 'none']:
            # XGBoost model is provided, construct the encoder
            return opts.files[0], None
        # only encoding file is provided
        return None, opts.files[0]
    
    def iswaxp(self, instance: Instance, 
              expl: Set[int]) -> bool:
        """
        Check if expl form a WAXp using SMT.
        
        Returns:
            True if expl form a WAXp
        """
        if isinstance(instance, tuple):
            instance, _ = instance

        # Convert to list format expected by XReason
        expl_list = sorted(list(expl))
        
        # Prepare or update the validator for this instance
        self._prepare_smt(instance, expl_list)
        
        # Check if satisfiable (not WAXp) or unsatisfiable (WAXp)
        # For WAXp: if UNSAT, then it's a WAXp
        is_sat = self.smt_checker.oracle.solve(
            [self.smt_checker.selv] + self.smt_checker.rhypos
        )
        
        return not is_sat  # UNSAT means WAXp
    
    def iswcxp(self, instance: Instance,
              expl: Set[int]) -> bool:
        """
        Check if expl form a WCXp using SMT.
        
        Returns:
            True if expl form a WCXp
        """
        if isinstance(instance, tuple):
            instance, _ = instance

        # For CXp, XReason uses complement
        # Get all feature indices from expl
        all_indices = set(self.smt_checker.ftids.values())
        complement = sorted(list(all_indices - set(expl)))
        
        # Prepare or update the validator with complement
        self._prepare_smt(instance, complement)
        
        # Check if satisfiable (WCXp) or unsatisfiable (not WCXp)
        # For WCXp: if SAT, then it's a WCXp
        hypos = [self.smt_checker.selv] + self.smt_checker.rhypos
        is_sat = self.smt_checker.oracle.solve(hypos)
        
        return is_sat  # SAT means WCXp

    def _prepare_smt(self, instance: np.ndarray, expl_list):
        sname = ','.join([str(v).strip() for v in instance])
        if sname not in self.smt_checker.idmgr.obj2id:
            self.smt_checker.prepare(instance, expl_list)
            return
        # else, update hypos
        existing = self.smt_checker.rhypos + getattr(self.smt_checker, "complement_hypos", [])
        sym_map = {s.symbol_name(): s for s in existing}

        base_hypos = []
        for inp in self.smt_checker.inps:
            feat = inp.symbol_name().split('_')[0]
            key = f"selv_{feat}"
            sym = sym_map.get(key)
            if sym is None:
                raise ValueError(f"Missing selector symbol for feature '{feat}'")
            base_hypos.append(sym)

        hypos = []
        complement_hypos = []
        for i, hypo in enumerate(base_hypos):
            j = self.smt_checker.ftids[self.smt_checker.xgb.transform_inverse_by_index(i)[0]]
            if j in expl_list:
                hypos.append(hypo)
            else:
                complement_hypos.append(hypo)
        
        self.smt_checker.rhypos = hypos
        self.smt_checker.complement_hypos = complement_hypos

        
    def init_smt(self):
        self.smt_checker = SMTValidator(
            self.xgb.enc, self.feature_names, len(self.target_names), self.xgb
        )        