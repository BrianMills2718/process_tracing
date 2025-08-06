#!/usr/bin/env python3
"""
Universal Unicode Fix for Windows - Long-term Solution

This module provides a comprehensive fix for Unicode encoding issues on Windows.
Import this at the start of any Python script to ensure proper UTF-8 handling.

Usage:
    import unicode_fix  # Automatically applies fixes
    
Or explicitly:
    from unicode_fix import fix_unicode_encoding
    fix_unicode_encoding()
"""

import os
import sys
import subprocess
import locale
import codecs


def fix_unicode_encoding():
    """
    Comprehensive Unicode fix for Windows systems.
    Addresses console output, subprocess calls, and file operations.
    """
    
    # Method 1: Environment variables (affects all subprocesses)
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    os.environ['PYTHONUTF8'] = '1'
    
    # Method 2: Windows console code page
    try:
        # Set console to UTF-8 (Windows 10+)
        os.system('chcp 65001 >nul 2>&1')
    except:
        pass
    
    # Method 3: Reconfigure stdout/stderr for current process
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
        except Exception:
            pass
    
    # Method 4: For older Python versions
    if sys.version_info < (3, 7):
        try:
            # Wrap stdout/stderr with UTF-8 codec
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
            sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())
        except Exception:
            pass
    
    # Method 5: Set default file encoding
    try:
        locale.setlocale(locale.LC_ALL, 'C.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')  
        except locale.Error:
            pass  # Keep system default if UTF-8 not available
    
    return True


def patch_subprocess():
    """
    Patch subprocess calls to use UTF-8 encoding by default.
    This fixes Unicode issues in subprocess.run(), Popen(), etc.
    """
    original_popen_init = subprocess.Popen.__init__
    
    def utf8_popen_init(self, *args, **kwargs):
        # Set encoding defaults for subprocess calls
        if 'encoding' not in kwargs and 'universal_newlines' not in kwargs:
            kwargs.setdefault('encoding', 'utf-8')
            kwargs.setdefault('errors', 'replace')
        return original_popen_init(self, *args, **kwargs)
    
    subprocess.Popen.__init__ = utf8_popen_init
    
    # Also patch subprocess.run for convenience
    original_run = subprocess.run
    
    def utf8_run(*args, **kwargs):
        kwargs.setdefault('encoding', 'utf-8')
        kwargs.setdefault('errors', 'replace')
        return original_run(*args, **kwargs)
    
    subprocess.run = utf8_run


def test_unicode_support():
    """Test Unicode output capability."""
    test_chars = [
        "✅ Success checkmark",
        "⚠️ Warning symbol", 
        "❌ Error symbol",
        "🎯 Target symbol",
        "📊 Chart symbol",
        "🔍 Magnifying glass",
        "⚡ Lightning bolt"
    ]
    
    print("Testing Unicode support:")
    for char in test_chars:
        try:
            print(f"  {char}")  
        except UnicodeEncodeError as e:
            print(f"  Failed: {e}")
            return False
    
    print("Unicode test completed successfully!")
    return True


def create_global_fix():
    """Create a site-packages fix for all Python projects."""
    try:
        import site
        site_packages = site.getsitepackages()[0]
        
        pth_content = """
# Unicode fix for Windows - Auto-applies to all Python scripts
import os, sys
os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
os.environ.setdefault('PYTHONUTF8', '1')
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except: pass
"""
        
        pth_file = os.path.join(site_packages, 'unicode_fix.pth')
        with open(pth_file, 'w', encoding='utf-8') as f:
            f.write(pth_content.strip())
        
        print(f"Global Unicode fix installed: {pth_file}")
        print("This will apply to all Python projects automatically.")
        return True
        
    except Exception as e:
        print(f"Could not install global fix: {e}")
        return False


# Automatically apply fixes when this module is imported
fix_unicode_encoding()
patch_subprocess()


if __name__ == "__main__":
    print("=== Unicode Fix for Windows ===")
    print()
    
    # Apply all fixes
    print("Applying Unicode fixes...")
    fix_unicode_encoding()
    patch_subprocess()
    
    # Test Unicode support
    if test_unicode_support():
        print("\n✅ Unicode fix successful!")
    else:
        print("\n❌ Unicode issues remain")
        print("Manual solutions to try:")
        print("  1. Run: chcp 65001")
        print("  2. Set environment: PYTHONIOENCODING=utf-8")
        print("  3. Use Windows Terminal instead of Command Prompt")
    
    # Offer to install global fix
    response = input("\nInstall global Unicode fix for all projects? (y/n): ")
    if response.lower().startswith('y'):
        create_global_fix()
    
    print("\nFor future projects, simply: import unicode_fix")