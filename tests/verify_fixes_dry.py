import sys
import os
import numpy as np

# Mocking heavy modules before they are imported by core.pipeline
from unittest.mock import MagicMock

sys.modules['torch'] = MagicMock()
sys.modules['diffusers'] = MagicMock()
sys.modules['transformers'] = MagicMock()
sys.modules['accelerate'] = MagicMock()
sys.modules['segment_anything'] = MagicMock()
sys.modules['insightface'] = MagicMock()
sys.modules['ultralytics'] = MagicMock()
sys.modules['compel'] = MagicMock()
sys.modules['controlnet_aux'] = MagicMock()

# Mocking some project modules that might be heavy
sys.modules['core.detector'] = MagicMock()
sys.modules['core.sam_wrapper'] = MagicMock()
sys.modules['core.model_manager'] = MagicMock()
sys.modules['core.face_restorer'] = MagicMock()
sys.modules['core.upscaler'] = MagicMock()
sys.modules['core.interrogator'] = MagicMock()

# Add project root to path
sys.path.append(os.getcwd())

def test_flag_logic():
    print("Testing Flag Logic in pipeline.py (Dry Run)...")
    # Read pipeline.py content to verify the fix manually via string check 
    # (since we can't easily import it with all its dependencies mocked without effort)
    with open('core/pipeline.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check bmab_enabled default
    if "if config.get('bmab_enabled', False):" in content:
        print("PASS: bmab_enabled default is now False.")
    else:
        print("FAIL: bmab_enabled default is NOT False.")
        # Print the line for debugging
        import re
        match = re.search(r"if config\.get\('bmab_enabled',.*\):", content)
        if match: print(f"Current line: {match.group(0)}")

    # Check color_fix logic
    if "if color_fix_method and color_fix_method != 'None':" in content:
        print("PASS: color_fix check is now robust.")
    else:
        print("FAIL: color_fix check is NOT robust.")

def test_upscaler_syntax():
    print("Testing Upscaler Syntax (Dry Run)...")
    with open('core/upscaler.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check scale initialization
    if "scale = getattr(self.model, 'scale', 4)" in content and "is_1x = (scale == 1)" in content:
        print("PASS: upscaler scale initialization fixed.")
    else:
        print("FAIL: upscaler scale initialization NOT fixed.")

if __name__ == "__main__":
    test_flag_logic()
    test_upscaler_syntax()
