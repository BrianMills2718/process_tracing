#!/usr/bin/env python3
"""
Universal Unicode Fix for Windows
This script sets up proper UTF-8 encoding for Python on Windows systems.
"""
import os
import sys

def fix_unicode_encoding():
    """Set up proper UTF-8 encoding for Windows."""
    
    # Method 1: Environment variables (most reliable)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Method 2: Reconfigure stdout/stderr for current session
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            print("✅ Unicode encoding configured successfully")
        except Exception as e:
            print(f"Warning: Could not reconfigure stdout/stderr: {e}")
    
    # Method 3: Set default encoding (older Python versions)
    try:
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
    except:
        pass  # Not critical if this fails
    
    return True

def create_unicode_startup_script():
    """Create a startup script to automatically fix Unicode issues."""
    startup_content = '''
import os
import sys

# Automatic Unicode fix for Windows
def _fix_unicode():
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except:
            pass

_fix_unicode()
'''
    
    with open('unicode_fix.py', 'w', encoding='utf-8') as f:
        f.write(startup_content.strip())
    
    print("Created unicode_fix.py - import this at the start of your scripts")

if __name__ == "__main__":
    fix_unicode_encoding()
    create_unicode_startup_script()
    
    # Test Unicode output
    try:
        print("✅ Success checkmark")
        print("⚠️ Warning symbol") 
        print("❌ Error symbol")
        print("🎯 Target symbol")
        print("Unicode test completed successfully!")
    except UnicodeEncodeError as e:
        print(f"Unicode test failed: {e}")
        print("Consider running: chcp 65001")
        print("Or setting PYTHONIOENCODING=utf-8 in environment")