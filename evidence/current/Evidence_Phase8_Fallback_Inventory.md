# Fallback Pattern Inventory

Generated: Fri, Aug 29, 2025  1:44:36 PM

## return None patterns
```
core/analyze.py:1437:    return None
core/analyze.py:2946:    if not results or 'metrics' not in results or 'node_type_distribution' not in results['metrics']: return None
core/analyze.py:2949:    if not node_dist_data: plt.close(); return None
core/analyze.py:2965:        return None
core/analyze.py:2968:    if not results or 'metrics' not in results or 'edge_type_distribution' not in results['metrics']: return None
core/analyze.py:2971:    if not edge_types_data: plt.close(); return None
core/analyze.py:2986:        return None
core/analyze.py:2990:        return None
core/analyze.py:2994:    if not path_nodes or len(path_nodes) < 1: return None
core/analyze.py:2998:        if subG.number_of_nodes() == 0: plt.close(); return None
core/analyze.py:3033:        return None
core/analyze.py:3036:    if not results or 'metrics' not in results or 'degree_centrality' not in results['metrics'] or not G: return None
core/analyze.py:3038:    if not centrality_data: plt.close(); return None
core/analyze.py:3066:        return None
core/analyze.py:3069:    if not results or 'evidence_analysis' not in results or not results['evidence_analysis']: return None
core/analyze.py:3081:    if not hypotheses_labels: plt.close(); return None
core/analyze.py:3106:        return None
core/diagnostic_rebalancer.py:155:                return None
core/diagnostic_rebalancer.py:186:        return None
core/enhance_evidence.py:42:            return None
core/enhance_evidence.py:87:        return None 
core/extract.py:467:    return None
core/extract.py:493:        return None
core/graph_alignment.py:635:            return None
core/llm_cache.py:199:                        return None
core/llm_cache.py:210:                        return None
core/llm_cache.py:232:                return None
core/mechanism_detector.py:331:            return None
core/mechanism_detector.py:352:            return None
core/mechanism_detector.py:399:            return None
```

## except with return patterns
```
core/analyze.py-578-        return value
core/analyze.py-1919-        return f"<div class='alert alert-danger'>Error generating network visualization: {e}</div>"
core/checkpoint.py-51-                return {}
core/disconnection_repair.py-607-            return {"new_edges": []}
core/enhance_evidence.py-42-            return None
core/extract.py:492:    except:
core/extract.py-493-        return None
core/extract.py-986-        return {'needs_repair': False, 'disconnection_rate': 0}
core/extraction_validator.py:233:    except Exception as e:
core/extraction_validator.py-234-        return {
core/llm_cache.py-232-                return None
core/llm_cache.py-295-                return False  
core/llm_cache.py-330-                return 0
core/llm_cache.py-371-                return 0
core/llm_reporting_utils.py-161-        return analysis_results
core/performance_profiler.py:81:        except:
core/performance_profiler.py-82-            return 0.0
core/plugins/advanced_van_evera_prediction_engine.py:664:                except ValueError:
core/plugins/advanced_van_evera_prediction_engine.py-665-                    return 0.0
core/plugins/advanced_van_evera_prediction_engine.py-886-            return self._create_default_evaluation_result()
core/plugins/bayesian_van_evera_engine.py-327-            return {'network_constructed': False, 'error': str(e)}
core/plugins/bayesian_van_evera_engine.py-432-            return {
core/plugins/dowhy_causal_analysis_engine.py-395-            return {'dowhy_model_created': False, 'error': str(e)}
core/plugins/dowhy_causal_analysis_engine.py-508-            return {
core/plugins/evidence_connector_enhancer.py-284-            return 0
core/plugins/evidence_connector_enhancer.py-343-            return 0
core/plugins/van_evera_testing.py-621-            return self._calculate_confidence_algorithmic(supporting, contradicting, prediction)
core/plugins/van_evera_testing.py-787-            return self._determine_overall_status_algorithmic(test_results, posterior)
core/plugins/van_evera_testing.py-990-            return self._calculate_confidence_interval_algorithmic(posterior, test_results)
core/plugin_integration.py-235-                return False
core/plugin_integration.py-242-        return False
core/streaming_html.py-602-                return {}
core/temporal_extraction.py:145:        except Exception:
core/temporal_extraction.py-146-            return None
core/temporal_extraction.py-393-            return {"temporal_expressions": [], "relationships": [], "sequences": []}
```

## hardcoded thresholds
```
```

## hardcoded thresholds in advanced_van_evera_prediction_engine.py
```
92:                    'quantitative_threshold': 0.70,
101:                    'quantitative_threshold': 0.60,
110:                    'quantitative_threshold': 0.65,
119:                    'quantitative_threshold': 0.50,
128:                    'quantitative_threshold': 0.75,
179:                    'quantitative_threshold': 0.65,
188:                    'quantitative_threshold': 0.55,
197:                    'quantitative_threshold': 0.60,
206:                    'quantitative_threshold': 0.70,
249:                    'quantitative_threshold': 0.60,
... (15 total instances)
```

## Priority Migration List

Based on the patterns found, here are the priority files for migration:

### Critical (Main execution path with fallbacks):
1. enhance_evidence.py - return None on LLM failure
2. diagnostic_rebalancer.py - return None patterns
3. temporal_extraction.py - exception handlers with return None

### High (Plugins with hardcoded values):
1. advanced_van_evera_prediction_engine.py - 15 hardcoded thresholds
2. van_evera_testing.py - fallback to algorithmic methods
3. evidence_connector_enhancer.py - return 0 on failures

### Medium (Supporting modules):
1. disconnection_repair.py - returns empty dict on failure
2. llm_cache.py - multiple return None patterns
3. mechanism_detector.py - return None patterns
