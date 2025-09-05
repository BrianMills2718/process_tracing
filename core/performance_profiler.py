"""
Performance Profiler for Process Tracing Analysis

Provides detailed timing and memory profiling with bottleneck identification
for the Van Evera Process Tracing Toolkit.

Author: Claude Code Implementation
Date: August 2025
"""

import time
import psutil
import threading
import json
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any
from functools import wraps
from dataclasses import dataclass, asdict
from contextlib import contextmanager
import traceback
import sys


@dataclass
class ProfileEntry:
    """Single profiling measurement entry"""
    phase: str
    start_time: float
    end_time: float
    duration: float
    memory_start: float
    memory_end: float
    memory_delta: float
    memory_peak: float
    success: bool
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ProfileReport:
    """Complete profiling report"""
    total_duration: float
    total_memory_peak: float
    total_memory_delta: float
    phases: List[ProfileEntry]
    bottlenecks: List[str]
    performance_score: float
    recommendations: List[str]


class PerformanceProfiler:
    """
    Advanced performance profiler with timing and memory tracking.
    
    Features:
    - Phase-based timing analysis
    - Memory usage tracking with peak detection
    - Bottleneck identification
    - Performance recommendations
    - Thread-safe operation
    - Detailed reporting
    """
    
    def __init__(self, enable_memory_monitoring: bool = True):
        self.phases: List[ProfileEntry] = []
        self.current_phase: Optional[str] = None
        self.phase_start_time: Optional[float] = None
        self.phase_start_memory: Optional[float] = None
        self.process = psutil.Process()
        self.enable_memory_monitoring = enable_memory_monitoring
        self.memory_samples: List[float] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False
        self._lock = threading.Lock()
        
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            return self.process.memory_info().rss / 1024 / 1024  # Convert to MB
        except:
            return 0.0
    
    def _memory_monitor(self):
        """Background thread for continuous memory monitoring"""
        while self.monitoring_active:
            try:
                memory_mb = self.get_memory_usage()
                with self._lock:
                    self.memory_samples.append(memory_mb)
                time.sleep(0.1)  # Sample every 100ms
            except:
                break
    
    def start_phase(self, phase_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Start profiling a new phase"""
        with self._lock:
            if self.current_phase:
                # Auto-end previous phase
                self.end_phase(f"auto_ended_{self.current_phase}")
            
            self.current_phase = phase_name
            self.phase_start_time = time.time()
            self.phase_start_memory = self.get_memory_usage()
            
            # Start memory monitoring if enabled
            if self.enable_memory_monitoring and not self.monitoring_active:
                self.memory_samples.clear()
                self.monitoring_active = True
                self.monitoring_thread = threading.Thread(target=self._memory_monitor, daemon=True)
                self.monitoring_thread.start()
    
    def end_phase(self, phase_name: Optional[str] = None, success: bool = True, 
                  error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """End profiling the current phase"""
        with self._lock:
            if not self.current_phase or not self.phase_start_time:
                return
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            duration = end_time - self.phase_start_time
            memory_delta = end_memory - self.phase_start_memory
            
            # Calculate peak memory during this phase
            memory_peak = max(self.memory_samples) if self.memory_samples else end_memory
            
            # Create profile entry
            entry = ProfileEntry(
                phase=phase_name or self.current_phase,
                start_time=self.phase_start_time,
                end_time=end_time,
                duration=duration,
                memory_start=self.phase_start_memory,
                memory_end=end_memory,
                memory_delta=memory_delta,
                memory_peak=memory_peak,
                success=success,
                error_message=error_message,
                metadata=metadata
            )
            
            self.phases.append(entry)
            
            # Reset for next phase
            self.current_phase = None
            self.phase_start_time = None
            self.phase_start_memory = None
            self.memory_samples.clear()
    
    def stop_monitoring(self):
        """Stop background memory monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=1.0)
    
    @contextmanager
    def profile_phase(self, phase_name: str, metadata: Optional[Dict[str, Any]] = None):
        """Context manager for profiling a phase"""
        self.start_phase(phase_name, metadata)
        try:
            yield
            self.end_phase(success=True, metadata=metadata)
        except Exception as e:
            self.end_phase(success=False, error_message=str(e), metadata=metadata)
            raise
    
    def profile_function(self, phase_name: Optional[str] = None, include_args: bool = False):
        """Decorator for profiling function calls"""
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                name = phase_name or f"{func.__module__}.{func.__name__}"
                metadata = {}
                
                if include_args:
                    # Safely capture arguments
                    try:
                        metadata['args_count'] = len(args)
                        metadata['kwargs_keys'] = list(kwargs.keys())
                    except:
                        pass
                
                with self.profile_phase(name, metadata):
                    return func(*args, **kwargs)
            return wrapper
        return decorator
    
    def get_bottlenecks(self, threshold_percent: float = 20.0) -> List[str]:
        """Identify performance bottlenecks"""
        if not self.phases:
            return []
        
        total_time = sum(phase.duration for phase in self.phases)
        if total_time == 0:
            return []
        
        bottlenecks = []
        for phase in self.phases:
            percentage = (phase.duration / total_time) * 100
            if percentage >= threshold_percent:
                bottlenecks.append(f"{phase.phase}: {phase.duration:.2f}s ({percentage:.1f}%)")
        
        return sorted(bottlenecks, key=lambda x: float(x.split(':')[1].split('s')[0]), reverse=True)
    
    def get_memory_bottlenecks(self, threshold_mb: float = 50.0) -> List[str]:
        """Identify memory usage bottlenecks"""
        bottlenecks = []
        for phase in self.phases:
            if abs(phase.memory_delta) >= threshold_mb:
                direction = "increased" if phase.memory_delta > 0 else "decreased"
                bottlenecks.append(f"{phase.phase}: {direction} by {abs(phase.memory_delta):.1f}MB")
        
        return bottlenecks
    
    def get_performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        if not self.phases:
            return 100.0
        
        # Base score
        score = 100.0
        
        # Deduct points for long durations
        total_time = sum(phase.duration for phase in self.phases)
        if total_time > 10:  # Target: <10s total
            score -= min(30, (total_time - 10) * 3)  # Max 30 points deduction
        
        # Deduct points for high memory usage
        max_memory = max(phase.memory_peak for phase in self.phases) if self.phases else 0
        if max_memory > 100:  # Target: <100MB
            score -= min(20, (max_memory - 100) / 10)  # Max 20 points deduction
        
        # Deduct points for failed phases
        failed_phases = sum(1 for phase in self.phases if not phase.success)
        score -= failed_phases * 25  # 25 points per failure
        
        return max(0.0, score)
    
    def get_recommendations(self) -> List[str]:
        """Generate performance improvement recommendations"""
        recommendations = []
        
        if not self.phases:
            return ["No profiling data available"]
        
        # Analyze total time
        total_time = sum(phase.duration for phase in self.phases)
        if total_time > 10:
            recommendations.append(f"Total analysis time ({total_time:.1f}s) exceeds 10s target - consider optimization")
        
        # Analyze memory usage
        max_memory = max(phase.memory_peak for phase in self.phases)
        if max_memory > 100:
            recommendations.append(f"Peak memory usage ({max_memory:.1f}MB) exceeds 100MB target - consider memory optimization")
        
        # Analyze bottlenecks
        bottlenecks = self.get_bottlenecks(15.0)  # 15% threshold
        if bottlenecks:
            recommendations.append(f"Top time bottleneck: {bottlenecks[0]}")
        
        memory_bottlenecks = self.get_memory_bottlenecks(30.0)  # 30MB threshold
        if memory_bottlenecks:
            recommendations.append(f"Memory concern: {memory_bottlenecks[0]}")
        
        # Analyze failures
        failed_phases = [phase for phase in self.phases if not phase.success]
        if failed_phases:
            recommendations.append(f"Failed phases detected: {[p.phase for p in failed_phases]}")
        
        # Suggest optimizations based on semantic phase analysis
        from core.llm_required import LLMRequiredError
        from core.plugins.van_evera_llm_interface import get_van_evera_llm
        
        llm_service = get_van_evera_llm()
        
        try:
            llm_phases = []
            
            for p in self.phases:
                # Use LLM to classify if this phase involves LLM processing
                classification_result = llm_service.assess_probative_value(
                    evidence_description=f"Performance phase: {p.phase}",
                    hypothesis_description="This phase involves LLM processing, AI inference, or text extraction operations",
                    context="Performance phase classification for optimization analysis"
                )
                
                if not hasattr(classification_result, 'probative_value'):
                    raise LLMRequiredError("Phase classification missing probative_value - invalid LLM response")
                    
                # High probability indicates LLM-related phase
                if classification_result.probative_value > 0.7:
                    llm_phases.append(p)
            
            if llm_phases and total_time > 0 and sum(p.duration for p in llm_phases) / total_time > 0.6:
                recommendations.append("LLM calls dominate execution time - consider implementing caching")
        except Exception as e:
            raise LLMRequiredError(f"Cannot classify performance phases without LLM: {e}")
        
        # Use LLM to classify HTML/formatting phases
        try:
            html_phases = []
            
            for p in self.phases:
                # Use LLM to classify if this phase involves HTML/formatting operations
                classification_result = llm_service.assess_probative_value(
                    evidence_description=f"Performance phase: {p.phase}",
                    hypothesis_description="This phase involves HTML generation, formatting, or presentation rendering operations",
                    context="Performance phase classification for HTML optimization analysis"
                )
                
                if not hasattr(classification_result, 'probative_value'):
                    raise LLMRequiredError("HTML phase classification missing probative_value - invalid LLM response")
                    
                # High probability indicates HTML/formatting-related phase
                if classification_result.probative_value > 0.7:
                    html_phases.append(p)
                    
            if html_phases and total_time > 0 and sum(p.duration for p in html_phases) / total_time > 0.3:
                recommendations.append("HTML generation takes significant time - consider streaming approach")
        except Exception as e:
            raise LLMRequiredError(f"Cannot classify HTML phases without LLM: {e}")
        
        if not recommendations:
            recommendations.append("Performance is within acceptable targets")
        
        return recommendations
    
    def generate_report(self) -> ProfileReport:
        """Generate comprehensive performance report"""
        total_duration = sum(phase.duration for phase in self.phases)
        total_memory_peak = max(phase.memory_peak for phase in self.phases) if self.phases else 0
        total_memory_delta = sum(phase.memory_delta for phase in self.phases)
        
        return ProfileReport(
            total_duration=total_duration,
            total_memory_peak=total_memory_peak,
            total_memory_delta=total_memory_delta,
            phases=self.phases,
            bottlenecks=self.get_bottlenecks(),
            performance_score=self.get_performance_score(),
            recommendations=self.get_recommendations()
        )
    
    def save_report(self, output_path: Path, include_raw_data: bool = True):
        """Save performance report to JSON file"""
        report = self.generate_report()
        
        # Convert to serializable format
        report_data = {
            'summary': {
                'total_duration': report.total_duration,
                'total_memory_peak': report.total_memory_peak,
                'total_memory_delta': report.total_memory_delta,
                'performance_score': report.performance_score,
                'bottlenecks': report.bottlenecks,
                'recommendations': report.recommendations
            }
        }
        
        if include_raw_data:
            report_data['phases'] = [asdict(phase) for phase in report.phases]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, default=str)
    
    def print_summary(self):
        """Print performance summary to console"""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("PERFORMANCE PROFILE SUMMARY")
        print("="*60)
        print(f"Total Duration: {report.total_duration:.2f}s")
        print(f"Peak Memory: {report.total_memory_peak:.1f}MB")
        print(f"Memory Delta: {report.total_memory_delta:+.1f}MB")
        print(f"Performance Score: {report.performance_score:.1f}/100")
        
        if report.bottlenecks:
            print(f"\nTop Bottlenecks:")
            for bottleneck in report.bottlenecks[:3]:
                print(f"  • {bottleneck}")
        
        if report.recommendations:
            print(f"\nRecommendations:")
            for rec in report.recommendations[:3]:
                print(f"  • {rec}")
        
        print(f"\nPhase Breakdown:")
        for phase in report.phases:
            status = "[OK]" if phase.success else "[FAIL]"
            print(f"  {status} {phase.phase}: {phase.duration:.2f}s ({phase.memory_delta:+.1f}MB)")
        
        print("="*60)
    
    def reset(self):
        """Reset profiler for new analysis"""
        self.stop_monitoring()
        with self._lock:
            self.phases.clear()
            self.current_phase = None
            self.phase_start_time = None
            self.phase_start_memory = None
            self.memory_samples.clear()


# Global profiler instance
_global_profiler: Optional[PerformanceProfiler] = None


def get_profiler() -> PerformanceProfiler:
    """Get or create global profiler instance"""
    global _global_profiler
    if _global_profiler is None:
        _global_profiler = PerformanceProfiler()
    return _global_profiler


def profile_phase(phase_name: str, metadata: Optional[Dict[str, Any]] = None):
    """Context manager for profiling a phase using global profiler"""
    return get_profiler().profile_phase(phase_name, metadata)


def profile_function(phase_name: Optional[str] = None, include_args: bool = False):
    """Decorator for profiling function calls using global profiler"""
    return get_profiler().profile_function(phase_name, include_args)


def print_performance_summary():
    """Print performance summary using global profiler"""
    get_profiler().print_summary()


def save_performance_report(output_path: Path, include_raw_data: bool = True):
    """Save performance report using global profiler"""
    get_profiler().save_report(output_path, include_raw_data)


def reset_profiler():
    """Reset global profiler"""
    global _global_profiler
    if _global_profiler:
        _global_profiler.reset()


if __name__ == "__main__":
    # Demo usage
    profiler = PerformanceProfiler()
    
    with profiler.profile_phase("demo_phase_1"):
        time.sleep(0.5)
        # Simulate some work
        data = [i**2 for i in range(100000)]
    
    with profiler.profile_phase("demo_phase_2"):
        time.sleep(0.3)
        # Simulate more work
        result = sum(data)
    
    profiler.print_summary()