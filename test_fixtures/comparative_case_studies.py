"""
Real-world comparative case studies for validation testing.

Contains realistic case study data based on actual political science and policy research
for validating Phase 5 comparative process tracing capabilities.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

class ComparativeCaseStudies:
    """Repository of real-world comparative case studies for validation."""
    
    @staticmethod
    def get_european_financial_crisis_cases() -> Dict[str, Dict[str, Any]]:
        """
        European Financial Crisis Response Cases (2008-2012)
        
        MSS Design: Similar institutional contexts (EU membership, democratic systems)
        with different policy responses and outcomes during the 2008 financial crisis.
        """
        return {
            "germany_2008": {
                "nodes": [
                    {
                        "id": "global_financial_crisis",
                        "type": "Event",
                        "description": "2008 global financial crisis hits German economy",
                        "timestamp": "2008-09-15T00:00:00",
                        "sequence_order": 1,
                        "properties": {
                            "severity": "high",
                            "external_shock": True,
                            "global_scope": True
                        }
                    },
                    {
                        "id": "coordinated_stimulus",
                        "type": "Mechanism",
                        "description": "Coordinated fiscal and monetary stimulus package",
                        "timestamp": "2008-11-01T00:00:00",
                        "sequence_order": 2,
                        "properties": {
                            "policy_type": "fiscal_monetary",
                            "coordination_level": "high",
                            "size_gdp_percent": 3.2
                        }
                    },
                    {
                        "id": "kurzarbeit_program",
                        "type": "Mechanism",
                        "description": "Short-time work (Kurzarbeit) employment program",
                        "timestamp": "2009-01-01T00:00:00",
                        "sequence_order": 3,
                        "properties": {
                            "policy_type": "labor_market",
                            "innovation": "institutional",
                            "scope": "national"
                        }
                    },
                    {
                        "id": "banking_stabilization",
                        "type": "Mechanism",
                        "description": "Banking sector stabilization measures",
                        "timestamp": "2008-12-01T00:00:00",
                        "sequence_order": 4,
                        "properties": {
                            "policy_type": "financial",
                            "intervention_type": "direct",
                            "scope": "sectoral"
                        }
                    },
                    {
                        "id": "economic_recovery",
                        "type": "Event",
                        "description": "Strong economic recovery and employment stability",
                        "timestamp": "2010-06-01T00:00:00",
                        "sequence_order": 5,
                        "properties": {
                            "outcome_type": "positive",
                            "recovery_speed": "fast",
                            "employment_impact": "minimal"
                        }
                    },
                    {
                        "id": "expert_assessment",
                        "type": "Evidence",
                        "description": "OECD assessment of German crisis response effectiveness",
                        "timestamp": "2011-01-01T00:00:00",
                        "sequence_order": 6,
                        "properties": {
                            "evidence_type": "expert_evaluation",
                            "source_credibility": "high",
                            "van_evera_type": "smoking_gun"
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "global_financial_crisis",
                        "target": "coordinated_stimulus",
                        "type": "triggers",
                        "strength": 0.9,
                        "mechanism": "policy_response",
                        "properties": {
                            "causal_logic": "crisis_response",
                            "time_lag_days": 47
                        }
                    },
                    {
                        "source": "global_financial_crisis",
                        "target": "kurzarbeit_program",
                        "type": "triggers",
                        "strength": 0.8,
                        "mechanism": "labor_protection",
                        "properties": {
                            "causal_logic": "employment_preservation",
                            "time_lag_days": 108
                        }
                    },
                    {
                        "source": "global_financial_crisis",
                        "target": "banking_stabilization",
                        "type": "triggers",
                        "strength": 0.85,
                        "mechanism": "financial_stability",
                        "properties": {
                            "causal_logic": "systemic_risk_mitigation",
                            "time_lag_days": 77
                        }
                    },
                    {
                        "source": "coordinated_stimulus",
                        "target": "economic_recovery",
                        "type": "produces",
                        "strength": 0.7,
                        "mechanism": "demand_stimulation",
                        "properties": {
                            "causal_logic": "keynesian_multiplier",
                            "time_lag_days": 213
                        }
                    },
                    {
                        "source": "kurzarbeit_program",
                        "target": "economic_recovery",
                        "type": "produces",
                        "strength": 0.8,
                        "mechanism": "employment_stability",
                        "properties": {
                            "causal_logic": "social_protection",
                            "time_lag_days": 517
                        }
                    },
                    {
                        "source": "banking_stabilization",
                        "target": "economic_recovery",
                        "type": "enables",
                        "strength": 0.6,
                        "mechanism": "credit_flow",
                        "properties": {
                            "causal_logic": "financial_intermediation",
                            "time_lag_days": 183
                        }
                    },
                    {
                        "source": "economic_recovery",
                        "target": "expert_assessment",
                        "type": "confirms",
                        "strength": 0.9,
                        "mechanism": "outcome_validation",
                        "properties": {
                            "causal_logic": "empirical_validation",
                            "time_lag_days": 214
                        }
                    }
                ],
                "metadata": {
                    "case_name": "German Financial Crisis Response 2008-2010",
                    "description": "Germany's coordinated policy response to the 2008 financial crisis",
                    "primary_outcome": "successful_recovery",
                    "secondary_outcomes": ["employment_preservation", "financial_stability", "export_competitiveness"],
                    "outcome_magnitude": 0.85,
                    "geographic_context": "Europe",
                    "institutional_context": "Democratic",
                    "economic_context": "Export-oriented market economy",
                    "political_context": "Federal parliamentary system",
                    "social_context": "Coordinated market economy",
                    "time_period": ["2008-09-15T00:00:00", "2010-12-31T23:59:59"],
                    "duration": "28 months",
                    "data_quality_score": 0.95,
                    "source_reliability": 0.9,
                    "evidence_completeness": 0.9,
                    "scope_conditions": ["EU_membership", "export_economy", "coordinated_capitalism"],
                    "key_actors": ["Federal Government", "Bundesbank", "Social Partners", "EU"],
                    "theoretical_framework": "Varieties of Capitalism",
                    "research_design": "MSS comparison"
                }
            },
            
            "greece_2010": {
                "nodes": [
                    {
                        "id": "sovereign_debt_crisis",
                        "type": "Event",
                        "description": "Greek sovereign debt crisis emerges",
                        "timestamp": "2010-04-01T00:00:00",
                        "sequence_order": 1,
                        "properties": {
                            "severity": "extreme",
                            "crisis_type": "sovereign_debt",
                            "external_pressure": True
                        }
                    },
                    {
                        "id": "troika_bailout",
                        "type": "Mechanism",
                        "description": "EU-ECB-IMF bailout program with conditionality",
                        "timestamp": "2010-05-02T00:00:00",
                        "sequence_order": 2,
                        "properties": {
                            "policy_type": "external_assistance",
                            "conditionality_level": "high",
                            "amount_billion_euros": 110
                        }
                    },
                    {
                        "id": "austerity_measures",
                        "type": "Mechanism",
                        "description": "Severe austerity measures and structural reforms",
                        "timestamp": "2010-06-01T00:00:00",
                        "sequence_order": 3,
                        "properties": {
                            "policy_type": "fiscal_contraction",
                            "severity": "extreme",
                            "gdp_impact_percent": -6.2
                        }
                    },
                    {
                        "id": "social_unrest",
                        "type": "Event",
                        "description": "Widespread social protests and political instability",
                        "timestamp": "2010-12-01T00:00:00",
                        "sequence_order": 4,
                        "properties": {
                            "event_type": "social_response",
                            "intensity": "high",
                            "political_impact": "destabilizing"
                        }
                    },
                    {
                        "id": "prolonged_recession",
                        "type": "Event",
                        "description": "Prolonged economic recession and high unemployment",
                        "timestamp": "2011-12-31T00:00:00",
                        "sequence_order": 5,
                        "properties": {
                            "outcome_type": "negative",
                            "duration": "extended",
                            "unemployment_rate": 24.3
                        }
                    },
                    {
                        "id": "academic_analysis",
                        "type": "Evidence",
                        "description": "Academic consensus on policy failure",
                        "timestamp": "2015-01-01T00:00:00",
                        "sequence_order": 6,
                        "properties": {
                            "evidence_type": "scholarly_consensus",
                            "source_credibility": "high",
                            "van_evera_type": "hoop"
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "sovereign_debt_crisis",
                        "target": "troika_bailout",
                        "type": "triggers",
                        "strength": 0.95,
                        "mechanism": "crisis_response",
                        "properties": {
                            "causal_logic": "external_intervention",
                            "time_lag_days": 31
                        }
                    },
                    {
                        "source": "troika_bailout",
                        "target": "austerity_measures",
                        "type": "requires",
                        "strength": 0.9,
                        "mechanism": "conditionality",
                        "properties": {
                            "causal_logic": "policy_conditionality",
                            "time_lag_days": 30
                        }
                    },
                    {
                        "source": "austerity_measures",
                        "target": "social_unrest",
                        "type": "triggers",
                        "strength": 0.8,
                        "mechanism": "social_backlash",
                        "properties": {
                            "causal_logic": "distributive_conflict",
                            "time_lag_days": 183
                        }
                    },
                    {
                        "source": "austerity_measures",
                        "target": "prolonged_recession",
                        "type": "produces",
                        "strength": 0.85,
                        "mechanism": "fiscal_contraction",
                        "properties": {
                            "causal_logic": "multiplier_effect",
                            "time_lag_days": 610
                        }
                    },
                    {
                        "source": "social_unrest",
                        "target": "prolonged_recession",
                        "type": "reinforces",
                        "strength": 0.6,
                        "mechanism": "political_instability",
                        "properties": {
                            "causal_logic": "confidence_effects",
                            "time_lag_days": 396
                        }
                    },
                    {
                        "source": "prolonged_recession",
                        "target": "academic_analysis",
                        "type": "confirms",
                        "strength": 0.9,
                        "mechanism": "outcome_evaluation",
                        "properties": {
                            "causal_logic": "empirical_assessment",
                            "time_lag_days": 1127
                        }
                    }
                ],
                "metadata": {
                    "case_name": "Greek Debt Crisis Response 2010-2015",
                    "description": "Greece's austerity-based response to the sovereign debt crisis",
                    "primary_outcome": "policy_failure",
                    "secondary_outcomes": ["social_disruption", "political_instability", "economic_contraction"],
                    "outcome_magnitude": 0.15,
                    "geographic_context": "Europe",
                    "institutional_context": "Democratic",
                    "economic_context": "Peripheral market economy",
                    "political_context": "Parliamentary system",
                    "social_context": "Mediterranean welfare state",
                    "time_period": ["2010-04-01T00:00:00", "2015-12-31T23:59:59"],
                    "duration": "69 months",
                    "data_quality_score": 0.9,
                    "source_reliability": 0.85,
                    "evidence_completeness": 0.8,
                    "scope_conditions": ["EU_membership", "peripheral_economy", "external_dependence"],
                    "key_actors": ["Greek Government", "Troika", "Social Movements", "EU"],
                    "theoretical_framework": "Varieties of Capitalism",
                    "research_design": "MSS comparison"
                }
            }
        }
    
    @staticmethod
    def get_covid_response_cases() -> Dict[str, Dict[str, Any]]:
        """
        COVID-19 Response Cases (2020-2021)
        
        MDS Design: Different institutional contexts (authoritarian vs democratic)
        with similar health outcomes through different policy mechanisms.
        """
        return {
            "south_korea_2020": {
                "nodes": [
                    {
                        "id": "covid_outbreak",
                        "type": "Event",
                        "description": "COVID-19 outbreak in South Korea",
                        "timestamp": "2020-02-18T00:00:00",
                        "sequence_order": 1,
                        "properties": {
                            "severity": "high",
                            "outbreak_source": "religious_gathering",
                            "initial_cases": 31
                        }
                    },
                    {
                        "id": "test_trace_isolate",
                        "type": "Mechanism",
                        "description": "Massive testing, contact tracing, and isolation",
                        "timestamp": "2020-02-20T00:00:00",
                        "sequence_order": 2,
                        "properties": {
                            "policy_type": "public_health",
                            "technology_use": "extensive",
                            "testing_capacity_daily": 20000
                        }
                    },
                    {
                        "id": "digital_surveillance",
                        "type": "Mechanism",
                        "description": "Digital contact tracing and movement monitoring",
                        "timestamp": "2020-02-25T00:00:00",
                        "sequence_order": 3,
                        "properties": {
                            "policy_type": "surveillance",
                            "technology_type": "digital",
                            "privacy_implications": "moderate"
                        }
                    },
                    {
                        "id": "transparent_communication",
                        "type": "Mechanism",
                        "description": "Transparent daily public communication",
                        "timestamp": "2020-02-22T00:00:00",
                        "sequence_order": 4,
                        "properties": {
                            "policy_type": "communication",
                            "transparency_level": "high",
                            "frequency": "daily"
                        }
                    },
                    {
                        "id": "outbreak_control",
                        "type": "Event",
                        "description": "Successful outbreak control without lockdown",
                        "timestamp": "2020-05-01T00:00:00",
                        "sequence_order": 5,
                        "properties": {
                            "outcome_type": "positive",
                            "lockdown_avoided": True,
                            "case_fatality_rate": 2.3
                        }
                    },
                    {
                        "id": "who_recognition",
                        "type": "Evidence",
                        "description": "WHO recognition of Korean model effectiveness",
                        "timestamp": "2020-06-01T00:00:00",
                        "sequence_order": 6,
                        "properties": {
                            "evidence_type": "international_recognition",
                            "source_credibility": "very_high",
                            "van_evera_type": "smoking_gun"
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "covid_outbreak",
                        "target": "test_trace_isolate",
                        "type": "triggers",
                        "strength": 0.95,
                        "mechanism": "public_health_response",
                        "properties": {
                            "causal_logic": "epidemic_control",
                            "time_lag_days": 2
                        }
                    },
                    {
                        "source": "covid_outbreak",
                        "target": "digital_surveillance",
                        "type": "triggers",
                        "strength": 0.8,
                        "mechanism": "contact_tracing",
                        "properties": {
                            "causal_logic": "transmission_tracking",
                            "time_lag_days": 7
                        }
                    },
                    {
                        "source": "covid_outbreak",
                        "target": "transparent_communication",
                        "type": "triggers",
                        "strength": 0.85,
                        "mechanism": "crisis_communication",
                        "properties": {
                            "causal_logic": "public_trust",
                            "time_lag_days": 4
                        }
                    },
                    {
                        "source": "test_trace_isolate",
                        "target": "outbreak_control",
                        "type": "produces",
                        "strength": 0.9,
                        "mechanism": "transmission_interruption",
                        "properties": {
                            "causal_logic": "epidemiological_intervention",
                            "time_lag_days": 71
                        }
                    },
                    {
                        "source": "digital_surveillance",
                        "target": "outbreak_control",
                        "type": "enables",
                        "strength": 0.7,
                        "mechanism": "contact_identification",
                        "properties": {
                            "causal_logic": "transmission_mapping",
                            "time_lag_days": 66
                        }
                    },
                    {
                        "source": "transparent_communication",
                        "target": "outbreak_control",
                        "type": "enables",
                        "strength": 0.6,
                        "mechanism": "public_compliance",
                        "properties": {
                            "causal_logic": "behavioral_change",
                            "time_lag_days": 69
                        }
                    },
                    {
                        "source": "outbreak_control",
                        "target": "who_recognition",
                        "type": "confirms",
                        "strength": 0.9,
                        "mechanism": "international_validation",
                        "properties": {
                            "causal_logic": "peer_recognition",
                            "time_lag_days": 32
                        }
                    }
                ],
                "metadata": {
                    "case_name": "South Korean COVID-19 Response 2020",
                    "description": "South Korea's technology-enabled COVID-19 containment strategy",
                    "primary_outcome": "outbreak_control",
                    "secondary_outcomes": ["economic_continuity", "international_recognition", "democratic_legitimacy"],
                    "outcome_magnitude": 0.9,
                    "geographic_context": "East Asia",
                    "institutional_context": "Democratic",
                    "economic_context": "Advanced market economy",
                    "political_context": "Presidential system",
                    "social_context": "High-tech society",
                    "time_period": ["2020-02-18T00:00:00", "2020-06-30T23:59:59"],
                    "duration": "4.4 months",
                    "data_quality_score": 0.95,
                    "source_reliability": 0.9,
                    "evidence_completeness": 0.85,
                    "scope_conditions": ["high_tech_capacity", "MERS_experience", "democratic_governance"],
                    "key_actors": ["KCDC", "Government", "Tech Companies", "Civil Society"],
                    "theoretical_framework": "State Capacity Theory",
                    "research_design": "MDS comparison"
                }
            },
            
            "china_2020": {
                "nodes": [
                    {
                        "id": "wuhan_outbreak",
                        "type": "Event",
                        "description": "COVID-19 outbreak in Wuhan, China",
                        "timestamp": "2019-12-31T00:00:00",
                        "sequence_order": 1,
                        "properties": {
                            "severity": "extreme",
                            "outbreak_location": "Wuhan",
                            "initial_response": "delayed"
                        }
                    },
                    {
                        "id": "strict_lockdown",
                        "type": "Mechanism",
                        "description": "Strict lockdown of Wuhan and Hubei Province",
                        "timestamp": "2020-01-23T00:00:00",
                        "sequence_order": 2,
                        "properties": {
                            "policy_type": "mobility_restriction",
                            "scope": "provincial",
                            "population_affected": 60000000
                        }
                    },
                    {
                        "id": "mass_surveillance",
                        "type": "Mechanism",
                        "description": "Comprehensive digital and physical surveillance",
                        "timestamp": "2020-01-25T00:00:00",
                        "sequence_order": 3,
                        "properties": {
                            "policy_type": "surveillance",
                            "technology_type": "comprehensive",
                            "privacy_implications": "extensive"
                        }
                    },
                    {
                        "id": "centralized_construction",
                        "type": "Mechanism",
                        "description": "Rapid construction of specialized hospitals",
                        "timestamp": "2020-01-24T00:00:00",
                        "sequence_order": 4,
                        "properties": {
                            "policy_type": "infrastructure",
                            "speed": "unprecedented",
                            "capacity_beds": 2300
                        }
                    },
                    {
                        "id": "outbreak_suppression",
                        "type": "Event",
                        "description": "Successful outbreak suppression in China",
                        "timestamp": "2020-04-08T00:00:00",
                        "sequence_order": 5,
                        "properties": {
                            "outcome_type": "positive",
                            "transmission_eliminated": True,
                            "case_fatality_rate": 4.0
                        }
                    },
                    {
                        "id": "international_scrutiny",
                        "type": "Evidence",
                        "description": "International debate over Chinese response effectiveness",
                        "timestamp": "2020-05-01T00:00:00",
                        "sequence_order": 6,
                        "properties": {
                            "evidence_type": "mixed_assessment",
                            "source_credibility": "variable",
                            "van_evera_type": "straw_in_the_wind"
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "wuhan_outbreak",
                        "target": "strict_lockdown",
                        "type": "triggers",
                        "strength": 0.9,
                        "mechanism": "authoritarian_response",
                        "properties": {
                            "causal_logic": "coercive_control",
                            "time_lag_days": 23
                        }
                    },
                    {
                        "source": "wuhan_outbreak",
                        "target": "mass_surveillance",
                        "type": "triggers",
                        "strength": 0.85,
                        "mechanism": "population_monitoring",
                        "properties": {
                            "causal_logic": "comprehensive_tracking",
                            "time_lag_days": 25
                        }
                    },
                    {
                        "source": "wuhan_outbreak",
                        "target": "centralized_construction",
                        "type": "triggers",
                        "strength": 0.8,
                        "mechanism": "state_mobilization",
                        "properties": {
                            "causal_logic": "resource_mobilization",
                            "time_lag_days": 24
                        }
                    },
                    {
                        "source": "strict_lockdown",
                        "target": "outbreak_suppression",
                        "type": "produces",
                        "strength": 0.85,
                        "mechanism": "transmission_blocking",
                        "properties": {
                            "causal_logic": "contact_elimination",
                            "time_lag_days": 75
                        }
                    },
                    {
                        "source": "mass_surveillance",
                        "target": "outbreak_suppression",
                        "type": "enables",
                        "strength": 0.7,
                        "mechanism": "compliance_enforcement",
                        "properties": {
                            "causal_logic": "behavioral_control",
                            "time_lag_days": 73
                        }
                    },
                    {
                        "source": "centralized_construction",
                        "target": "outbreak_suppression",
                        "type": "enables",
                        "strength": 0.6,
                        "mechanism": "treatment_capacity",
                        "properties": {
                            "causal_logic": "healthcare_provision",
                            "time_lag_days": 74
                        }
                    },
                    {
                        "source": "outbreak_suppression",
                        "target": "international_scrutiny",
                        "type": "triggers",
                        "strength": 0.8,
                        "mechanism": "legitimacy_debate",
                        "properties": {
                            "causal_logic": "normative_assessment",
                            "time_lag_days": 23
                        }
                    }
                ],
                "metadata": {
                    "case_name": "Chinese COVID-19 Response 2020",
                    "description": "China's authoritarian lockdown approach to COVID-19 containment",
                    "primary_outcome": "outbreak_control",
                    "secondary_outcomes": ["authoritarian_efficiency", "international_controversy", "economic_disruption"],
                    "outcome_magnitude": 0.85,
                    "geographic_context": "East Asia",
                    "institutional_context": "Authoritarian",
                    "economic_context": "State-led market economy",
                    "political_context": "Single-party system",
                    "social_context": "High state capacity",
                    "time_period": ["2019-12-31T00:00:00", "2020-05-31T23:59:59"],
                    "duration": "5 months",
                    "data_quality_score": 0.8,
                    "source_reliability": 0.7,
                    "evidence_completeness": 0.75,
                    "scope_conditions": ["authoritarian_regime", "high_state_capacity", "centralized_system"],
                    "key_actors": ["CCP", "Local Governments", "Military", "Healthcare System"],
                    "theoretical_framework": "State Capacity Theory",
                    "research_design": "MDS comparison"
                }
            }
        }
    
    @staticmethod
    def get_democratization_cases() -> Dict[str, Dict[str, Any]]:
        """
        Democratic Transition Cases (1980s-1990s)
        
        Diverse Case Design: Multiple pathways to democratization
        for testing pattern detection across different contexts.
        """
        return {
            "spain_transition": {
                "nodes": [
                    {
                        "id": "franco_death",
                        "type": "Event",
                        "description": "Death of Francisco Franco",
                        "timestamp": "1975-11-20T00:00:00",
                        "sequence_order": 1,
                        "properties": {
                            "regime_change": "leadership_succession",
                            "uncertainty": "high",
                            "international_pressure": "moderate"
                        }
                    },
                    {
                        "id": "elite_consensus",
                        "type": "Mechanism",
                        "description": "Elite consensus on democratic transition",
                        "timestamp": "1976-06-01T00:00:00",
                        "sequence_order": 2,
                        "properties": {
                            "mechanism_type": "elite_bargaining",
                            "inclusiveness": "broad",
                            "time_horizon": "long_term"
                        }
                    },
                    {
                        "id": "gradual_liberalization",
                        "type": "Mechanism",
                        "description": "Gradual political liberalization",
                        "timestamp": "1976-07-01T00:00:00",
                        "sequence_order": 3,
                        "properties": {
                            "pace": "gradual",
                            "scope": "comprehensive",
                            "reversibility": "low"
                        }
                    },
                    {
                        "id": "constitutional_referendum",
                        "type": "Event",
                        "description": "Constitutional referendum approval",
                        "timestamp": "1978-12-06T00:00:00",
                        "sequence_order": 4,
                        "properties": {
                            "legitimation_method": "referendum",
                            "approval_rate": 87.9,
                            "participation": "high"
                        }
                    },
                    {
                        "id": "democratic_consolidation",
                        "type": "Event",
                        "description": "Successful democratic consolidation",
                        "timestamp": "1982-10-28T00:00:00",
                        "sequence_order": 5,
                        "properties": {
                            "outcome_type": "consolidation",
                            "stability": "high",
                            "legitimacy": "broad"
                        }
                    }
                ],
                "edges": [
                    {
                        "source": "franco_death",
                        "target": "elite_consensus",
                        "type": "enables",
                        "strength": 0.8,
                        "mechanism": "opportunity_structure"
                    },
                    {
                        "source": "elite_consensus",
                        "target": "gradual_liberalization",
                        "type": "produces",
                        "strength": 0.9,
                        "mechanism": "negotiated_change"
                    },
                    {
                        "source": "gradual_liberalization",
                        "target": "constitutional_referendum",
                        "type": "enables",
                        "strength": 0.85,
                        "mechanism": "institutional_foundation"
                    },
                    {
                        "source": "constitutional_referendum",
                        "target": "democratic_consolidation",
                        "type": "produces",
                        "strength": 0.9,
                        "mechanism": "legitimation"
                    }
                ],
                "metadata": {
                    "case_name": "Spanish Democratic Transition 1975-1982",
                    "description": "Spain's negotiated transition from Francoist dictatorship",
                    "primary_outcome": "democratic_consolidation",
                    "outcome_magnitude": 0.9,
                    "geographic_context": "Southern Europe",
                    "institutional_context": "Transition",
                    "time_period": ["1975-11-20T00:00:00", "1982-10-28T23:59:59"],
                    "theoretical_framework": "Transition Theory",
                    "research_design": "Diverse case comparison"
                }
            }
        }

    @staticmethod
    def create_test_files(output_directory: str) -> List[str]:
        """
        Create JSON files for all comparative case studies.
        
        Args:
            output_directory: Directory to write test case files
            
        Returns:
            List of created file paths
        """
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        
        created_files = []
        
        # European Financial Crisis Cases
        crisis_cases = ComparativeCaseStudies.get_european_financial_crisis_cases()
        for case_id, case_data in crisis_cases.items():
            file_path = output_path / f"{case_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, indent=2, ensure_ascii=False)
            created_files.append(str(file_path))
        
        # COVID-19 Response Cases
        covid_cases = ComparativeCaseStudies.get_covid_response_cases()
        for case_id, case_data in covid_cases.items():
            file_path = output_path / f"{case_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, indent=2, ensure_ascii=False)
            created_files.append(str(file_path))
        
        # Democratization Cases
        demo_cases = ComparativeCaseStudies.get_democratization_cases()
        for case_id, case_data in demo_cases.items():
            file_path = output_path / f"{case_id}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(case_data, f, indent=2, ensure_ascii=False)
            created_files.append(str(file_path))
        
        return created_files
    
    @staticmethod
    def get_expected_patterns() -> Dict[str, Any]:
        """
        Return expected patterns for validation testing.
        
        Returns:
            Dictionary of expected patterns for comparative analysis validation
        """
        return {
            "mss_patterns": {
                "european_crisis": {
                    "similar_contexts": ["EU_membership", "democratic_institutions", "market_economy"],
                    "different_outcomes": ["successful_recovery", "prolonged_recession"],
                    "causal_factors": ["policy_coordination", "institutional_capacity", "external_constraints"],
                    "expected_similarity": 0.8
                }
            },
            "mds_patterns": {
                "covid_response": {
                    "different_contexts": ["democratic_vs_authoritarian", "technology_adoption", "state_capacity"],
                    "similar_outcomes": ["outbreak_control", "public_health_success"],
                    "common_mechanisms": ["rapid_response", "surveillance", "compliance"],
                    "expected_similarity": 0.3
                }
            },
            "recurring_mechanisms": [
                {
                    "pattern_name": "Crisis Response Pattern",
                    "description": "Government response to external shocks",
                    "core_sequence": ["external_shock", "policy_response", "outcome"],
                    "variations": ["coordinated_vs_unilateral", "fiscal_vs_monetary", "democratic_vs_authoritarian"],
                    "frequency": 0.8
                },
                {
                    "pattern_name": "Elite Bargaining Pattern",
                    "description": "Elite consensus formation during transitions",
                    "core_sequence": ["political_opening", "elite_negotiation", "institutional_change"],
                    "variations": ["gradual_vs_rapid", "inclusive_vs_exclusive"],
                    "frequency": 0.6
                }
            ]
        }