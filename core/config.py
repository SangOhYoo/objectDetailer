import os
import yaml

class AppConfig:
    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.load_config(config_path)
        return cls._instance

    def load_config(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Configuration file not found: {path}")
            
        with open(path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f)
            
        print(f"[Config] Loaded configuration from {path}")

    # core/config.py 보강

    def get_path(self, category, file_key=None):
        if not self.data or 'paths' not in self.data:
            print("[Config] Error: 'paths' section missing in config.yaml")
            return "" # Return empty string to handle gracefully
            
        base_dir = self.data['paths'].get(category, "")
        if not base_dir: return ""
            
        if file_key:
            filename = self.data.get('files', {}).get(file_key, "")
            if filename:
                return os.path.join(base_dir, filename)
        
        return base_dir

    def get(self, *keys):
        """중첩된 키 값 가져오기 (예: get('defaults', 'denoise'))"""
        val = self.data
        for k in keys:
            val = val.get(k)
            if val is None: return None
        return val

# 싱글톤 인스턴스 (어디서든 import config_instance 로 사용)
config_instance = AppConfig()