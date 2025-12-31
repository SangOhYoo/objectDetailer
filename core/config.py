import os
import yaml

class AppConfig:
    _instance = None

    def __new__(cls, config_path="config.yaml"):
        if cls._instance is None:
            cls._instance = super(AppConfig, cls).__new__(cls)
            cls._instance.config_path = config_path
            cls._instance.load_config(config_path)
        return cls._instance

    def load_config(self, path):
        if not os.path.exists(path):
            # 파일이 없으면 빈 딕셔너리로 초기화하거나 기본값 생성
            self.data = {}
            print(f"[Config] Warning: Configuration file not found at {path}. Starting with empty config.")
            return
            
        with open(path, 'r', encoding='utf-8') as f:
            self.data = yaml.safe_load(f) or {}
            
        print(f"[Config] Loaded configuration from {path}")

    def save_config(self, new_data=None):
        """현재 설정(혹은 새로운 데이터)을 yaml 파일에 저장"""
        if new_data:
            self.data.update(new_data)
            
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            print(f"[Config] Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"[Config] Error saving configuration: {e}")
            return False

    def get_path(self, category, file_key=None):
        if not self.data or 'paths' not in self.data:
            # 경로 설정이 없을 경우를 대비한 방어 코드
            return ""
            
        base_dir = self.data['paths'].get(category, "")
        if not base_dir: return ""
            
        if file_key:
            filename = self.data.get('files', {}).get(file_key, "")
            if filename:
                return os.path.join(base_dir, filename)
        
        return base_dir

    def get(self, *keys):
        """중첩된 키 값 가져오기"""
        val = self.data
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return None
            if val is None: return None
        return val

config_instance = AppConfig()