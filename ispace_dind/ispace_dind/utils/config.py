from pathlib import Path
import yaml

class YamlConfig:
    def __init__(self, config_dir, config_name):
        self.config_dir = Path(config_dir)

        if config_name.endswith(".yaml"):
            self.file_name = config_name
            self.root_name = Path(config_name).stem
        else:
            self.file_name = config_name + ".yaml"
            self.root_name = config_name

        self.config_path = self.config_dir / self.file_name
        self._data = {}

        self.config_dir.mkdir(parents=True, exist_ok=True)

        if self.config_path.exists():
            self._load()
        else:
            self._save()

    def _load(self):
        with open(self.config_path, "r", encoding="utf-8") as f:
            self._data = yaml.safe_load(f) or {}

    def _save(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self._data, f, allow_unicode=True, sort_keys=False)

    def save(self):
        self._save()

    def reload(self):
        self._load()

    def add_default(self, key_path, value):
        keys = key_path.split(".")
        current = self._data

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                return
            current = current[key]

        last_key = keys[-1]
        if last_key not in current:
            current[last_key] = value
            self._save()

    def set(self, key_path, value):
        keys = key_path.split(".")
        current = self._data

        for key in keys[:-1]:
            if key not in current or not isinstance(current[key], dict):
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value
        self._save()

    def get(self, key_path, default=None):
        keys = key_path.split(".")
        current = self._data

        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]

        return current

    def get_dict(self):
        result = {}
        self._flatten_dict(self._data, self.root_name, result)
        return result

    def _flatten_dict(self, current, prefix, result):
        for key, value in current.items():
            new_key = f"{prefix}.{key}"
            if isinstance(value, dict):
                self._flatten_dict(value, new_key, result)
            else:
                result[new_key] = value