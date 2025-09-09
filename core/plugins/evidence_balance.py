"""
Evidence Balance Plugin
Fixes evidence balance math error (#16) with correct probative value calculation
"""
import logging
from typing import Any, Dict, List, Union

from .base import ProcessTracingPlugin, PluginValidationError


class EvidenceBalancePlugin(ProcessTracingPlugin):
    """Correctly calculates evidence balance using proper probative value math"""
    
    plugin_id = "evidence_balance"
    
    def validate_input(self, data: Any) -> None:
        """
        Validate evidence balance calculation input.
        
        Args:
            data: Dictionary with evidence and hypothesis data
            
        Raises:
            PluginValidationError: If input is invalid
        """
        if not isinstance(data, dict):
            raise PluginValidationError(
                self.id,
                f"Input must be dictionary, got {type(data)}"
            )
        
        required_keys = ['hypothesis', 'evidence_list']
        for key in required_keys:
            if key not in data:
                raise PluginValidationError(
                    self.id,
                    f"Missing required key '{key}' in input data"
                )
        
        hypothesis = data['hypothesis']
        if not isinstance(hypothesis, dict):
            raise PluginValidationError(
                self.id,
                f"Hypothesis must be dictionary, got {type(hypothesis)}"
            )
        
        if 'balance' not in hypothesis:
            raise PluginValidationError(
                self.id,
                "Hypothesis must have 'balance' key"
            )
        
        try:
            float(hypothesis['balance'])
        except (ValueError, TypeError):
            raise PluginValidationError(
                self.id,
                f"Hypothesis balance must be numeric, got {hypothesis['balance']}"
            )
        
        evidence_list = data['evidence_list']
        if not isinstance(evidence_list, list):
            raise PluginValidationError(
                self.id,
                f"Evidence list must be list, got {type(evidence_list)}"
            )
        
        # Validate each evidence item
        for i, evidence in enumerate(evidence_list):
            if not isinstance(evidence, dict):
                raise PluginValidationError(
                    self.id,
                    f"Evidence item {i} must be dictionary, got {type(evidence)}"
                )
            
            if 'probative_value' not in evidence:
                raise PluginValidationError(
                    self.id,
                    f"Evidence item {i} missing 'probative_value'"
                )
            
            try:
                float(evidence['probative_value'])
            except (ValueError, TypeError):
                raise PluginValidationError(
                    self.id,
                    f"Evidence item {i} probative_value must be numeric, got {evidence['probative_value']}"
                )
    
    def execute(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate evidence balance with correct probative value math.
        
        Args:
            data: Dictionary with hypothesis and evidence data
            
        Returns:
            Dictionary with updated hypothesis balance and calculation details
        """
        self.logger.info("START: Evidence balance calculation")
        
        hypothesis = data['hypothesis'].copy()  # Don't modify original
        evidence_list = data['evidence_list']
        
        original_balance = float(hypothesis['balance'])
        self.logger.info(f"PROGRESS: Initial hypothesis balance: {original_balance}")
        
        # Calculate new balance using CORRECT math (no -abs() bug)
        balance_changes = []
        running_balance = original_balance
        
        for i, evidence in enumerate(evidence_list):
            probative_value = float(evidence['probative_value'])
            
            # FIXED: Use probative_value directly (no -abs() wrapper)
            balance_effect = probative_value
            
            running_balance += balance_effect
            
            change_record = {
                'evidence_id': evidence.get('id', f'evidence_{i}'),
                'description': evidence.get('description', 'Unknown evidence'),
                'probative_value': probative_value,
                'balance_effect': balance_effect,
                'running_balance': running_balance
            }
            balance_changes.append(change_record)
            
            direction = "increases" if balance_effect > 0 else "decreases" if balance_effect < 0 else "maintains"
            self.logger.info(f"PROGRESS: Evidence {i+1} {direction} balance by {balance_effect:.3f} (new total: {running_balance:.3f})")
        
        # Update hypothesis balance
        final_balance = running_balance
        hypothesis['balance'] = final_balance
        
        # Calculate summary statistics
        positive_evidence = [e for e in evidence_list if float(e['probative_value']) > 0]
        negative_evidence = [e for e in evidence_list if float(e['probative_value']) < 0]
        neutral_evidence = [e for e in evidence_list if float(e['probative_value']) == 0]
        
        total_positive = sum(float(e['probative_value']) for e in positive_evidence)
        total_negative = sum(float(e['probative_value']) for e in negative_evidence)
        net_effect = final_balance - original_balance
        
        stats = {
            'original_balance': original_balance,
            'final_balance': final_balance,
            'net_effect': net_effect,
            'total_evidence': len(evidence_list),
            'positive_evidence_count': len(positive_evidence),
            'negative_evidence_count': len(negative_evidence),
            'neutral_evidence_count': len(neutral_evidence),
            'total_positive_effect': total_positive,
            'total_negative_effect': total_negative
        }
        
        self.logger.info(f"PROGRESS: Balance calculation complete - {original_balance:.3f} → {final_balance:.3f} (Δ{net_effect:+.3f})")
        self.logger.info(f"PROGRESS: Evidence breakdown: {len(positive_evidence)} positive, {len(negative_evidence)} negative, {len(neutral_evidence)} neutral")
        self.logger.info("END: Evidence balance calculation completed successfully")
        
        return {
            'hypothesis': hypothesis,
            'balance_changes': balance_changes,
            'calculation_stats': stats,
            'evidence_summary': {
                'positive_evidence': positive_evidence,
                'negative_evidence': negative_evidence,
                'neutral_evidence': neutral_evidence
            }
        }
    
    def get_checkpoint_data(self) -> Dict[str, Any]:
        """Return checkpoint data for evidence balance calculation."""
        return {
            'plugin_id': self.id,
            'stage': 'evidence_balance',
            'status': 'completed'
        }