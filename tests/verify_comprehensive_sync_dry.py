import os

def check_file(path, search_terms):
    if not os.path.exists(path):
        print(f"FAILED: {path} not found")
        return False
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    all_found = True
    for term in search_terms:
        if term in content:
            print(f"PASSED: Found '{term[:50]}...' in {os.path.basename(path)}")
        else:
            print(f"FAILED: Could NOT find '{term}' in {os.path.basename(path)}")
            all_found = False
    return all_found

# 1. Pipeline logic checks
pipeline_path = r"d:\SAM3_FaceDetailer_Ultimate\core\pipeline.py"
pipeline_terms = [
    "if config.get('ignore_edge_touching', False):",
    "if x1 <= 2 or y1 <= 2 or x2 >= w - 2 or y2 >= h - 2:",
    "if config.get('auto_prompt_injection', True):",
    "quality_pos = \"high quality, detailed, masterpiece, 8k\"",
    "if ui_w > 0 or ui_h > 0:",
    "target_res = max(ui_w, ui_h)",
    "inpaint_area_whole = config.get('inpaint_full_res', False)",
    "if inpaint_area_whole:",
    "if config.get('sep_noise', False) and seed != -1:",
    "seed = seed + int(box[0] + box[1])"
]

# 2. Worker logic checks
worker_path = r"d:\SAM3_FaceDetailer_Ultimate\ui\workers.py"
worker_terms = [
    "if cfg.get('system', 'save_metadata', True):",
    "save_image_with_metadata(result_img, fpath, save_path, ConfigWrapper(active_conf))",
    "from core.io_utils import imwrite",
    "imwrite(save_path, result_img)"
]

# 3. UI logic checks
ui_path = r"d:\SAM3_FaceDetailer_Ultimate\ui\main_window_tabs.py"
ui_terms = [
    "self.chk_auto_prompt = QCheckBox(\"✨ 품질 보정 (Quality)\")",
    "self.chk_auto_prompt.setChecked(self.saved_config.get('auto_prompt_injection', True))"
]

print("--- UI-Logic Sync Verification (Dry Run) ---")
v1 = check_file(pipeline_path, pipeline_terms)
v2 = check_file(worker_path, worker_terms)
v3 = check_file(ui_path, ui_terms)

if v1 and v2 and v3:
    print("\nOVERALL STATUS: ALL FIXES VERIFIED IN SOURCE CODE.")
else:
    print("\nOVERALL STATUS: SOME FIXES MISSING OR MISMATCHED.")
    exit(1)
