# PyGent 3D Studio Configuration
from pathlib import Path

class ThreestudioConfig:
    BATCH_SIZE = 2
    OUTPUT_FORMATS = ['glb', 'obj', 'stl', 'ply']
    SERVER_PORT = 8003
    # Generation methods with detailed configs
    METHODS = {
        'dreamfusion': {'config': 'configs/dreamfusion.yaml', 'vram': '6-8GB'},
        'zero123': {'config': 'configs/zero123.yaml', 'vram': '4-6GB'}
    }
