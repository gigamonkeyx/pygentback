"""
Create Deployment Files

Creates the essential files needed for GitHub repository and Cloudflare Pages deployment.
"""

import os
import shutil
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_deployment_files():
    """Create deployment files."""
    logger.info("Creating deployment files...")
    
    # Create deployment directory
    deployment_path = Path(__file__).parent / "deployment_ready"
    ui_path = Path(__file__).parent / "ui"
    
    if deployment_path.exists():
        shutil.rmtree(deployment_path)
    deployment_path.mkdir()
    
    # Copy UI files
    if ui_path.exists():
        shutil.copytree(ui_path, deployment_path, dirs_exist_ok=True)
        logger.info("UI files copied")
    
    # Update package.json
    package_json_path = deployment_path / "package.json"
    if package_json_path.exists():
        with open(package_json_path, 'r', encoding='utf-8') as f:
            package_data = json.load(f)
        
        package_data.update({
            "name": "pygent-factory-ui",
            "homepage": "https://timpayne.net/pygent",
            "repository": {
                "type": "git",
                "url": "https://github.com/gigamonkeyx/pygent.git"
            }
        })
        
        with open(package_json_path, 'w', encoding='utf-8') as f:
            json.dump(package_data, f, indent=2)
        logger.info("package.json updated")
    
    # Create simple README
    readme_content = """# PyGent Factory UI

Advanced AI Reasoning System - Web Interface

## Live Demo

Visit: https://timpayne.net/pygent

## Features

- Multi-Agent Chat Interface
- Tree of Thought Reasoning
- System Monitoring
- MCP Marketplace
- Zero Mock Code Integration

## Technology Stack

- React 18 + TypeScript
- Zustand State Management
- Radix UI + Tailwind CSS
- WebSocket Real-time Communication
- Vite Build Tool
- Cloudflare Pages Deployment

## Development

```bash
npm install
npm run dev
```

## Build

```bash
npm run build
```

## Deployment

Deployed to Cloudflare Pages with custom domain routing.

Backend services run locally and connect through Cloudflare Tunnels.
"""
    
    readme_path = deployment_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    logger.info("README created")
    
    # Create deployment instructions
    instructions_content = """# Deployment Instructions

## Step 1: GitHub Repository

1. Clone repository:
   ```
   git clone https://github.com/gigamonkeyx/pygent.git
   cd pygent
   ```

2. Copy files from deployment_ready/ to repository root

3. Commit and push:
   ```
   git add .
   git commit -m "Add PyGent Factory UI"
   git push origin main
   ```

## Step 2: Cloudflare Pages

1. Go to Cloudflare Pages dashboard
2. Connect GitHub repository: gigamonkeyx/pygent
3. Configure build settings:
   - Build command: npm run build
   - Build output: dist
   - Environment variables:
     - VITE_API_BASE_URL=https://api.timpayne.net
     - VITE_WS_BASE_URL=wss://ws.timpayne.net

## Step 3: Custom Domain

1. Add custom domain: timpayne.net
2. Configure subdirectory routing: /pygent

## Step 4: Backend Tunnels

1. Install cloudflared
2. Create tunnel for backend services
3. Configure DNS routing

## Success Criteria

- UI accessible at https://timpayne.net/pygent
- WebSocket connections functional
- Real-time features working
- Agent responses displaying correctly
"""
    
    instructions_path = deployment_path / "DEPLOYMENT.md"
    with open(instructions_path, 'w', encoding='utf-8') as f:
        f.write(instructions_content)
    logger.info("Deployment instructions created")
    
    logger.info(f"Deployment files ready at: {deployment_path}")
    return True


if __name__ == "__main__":
    success = create_deployment_files()
    logger.info("Deployment preparation complete!" if success else "Deployment preparation failed!")