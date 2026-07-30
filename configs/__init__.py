import os
import yaml
import re
from typing import Dict, Any


class ConfigLoader:
    """配置加载器 - 从 YAML 文件加载配置"""

    def __init__(self, config_path: str = None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), 'config.yaml'
        )
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载 YAML 配置文件，支持变量替换"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            content = f.read()

        while True:
            # 匹配 ${variable} 模式
            matches = re.findall(r'\$\{([^}]+)\}', content)
            if not matches:
                break
            raw_config = yaml.safe_load(content)
            replaced = False
            for match in matches:
                # 获取变量路径（支持点分隔，如 data_dirs.info）
                value = self._get_nested_value(raw_config, match)
                if value:
                    content = content.replace(f'${{{match}}}', str(value))
                    replaced = True

            if not replaced:
                break
        return yaml.safe_load(content)
    def _get_nested_value(self, config: Dict, path: str):
        """获取嵌套字典中的值，支持点分隔路径"""
        keys = path.split('.')
        value = config
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            # 如果找不到变量，检查环境变量
            return os.environ.get(path)
    def get(self, key: str, default=None):
        """获取配置值，支持点分隔路径"""
        keys = key.split('.')
        current = self.config
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current


# 全局配置实例
config = ConfigLoader()
