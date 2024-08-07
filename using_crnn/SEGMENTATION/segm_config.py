import json
import os

class Config:
    def __init__(self, config_path):
        with open(os.path.join(config_path), 'r') as f:
            self.config = json.load(f)

    def get_classes(self):
        return self.config['classes']

    def get_image(self, key):
        return self.config['image'][key]

