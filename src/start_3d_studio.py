# PyGent 3D Studio Startup Script
from studio_3d_main import PyGent3DStudio
from studio_3d_config import ThreestudioConfig

if __name__ == '__main__':
    print('Starting PyGent 3D Studio...')
    studio = PyGent3DStudio()
    print(f'3D Studio ready on port {ThreestudioConfig.SERVER_PORT}')
