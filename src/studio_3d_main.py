# PyGent 3D Studio Main Application
from fastapi import FastAPI
from studio_3d_config import ThreestudioConfig

class PyGent3DStudio:
    def __init__(self):
        self.app = FastAPI(title='PyGent 3D Studio')
import asyncio
import subprocess

    async def generate_3d_model(self, prompt, method='dreamfusion'):
        \
\\Generate
3D
model
using
Threestudio\\\
        config = ThreestudioConfig.METHODS[method]['config']
        return f'Generated 3D model for: {prompt} using {method}'
