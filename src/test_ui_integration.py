"""
UI Integration Test

Tests the UI setup and integration with backend services.
"""

import asyncio
import logging
import subprocess
import sys
import os
import time
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UIIntegrationTester:
    """Tests UI integration with PyGent Factory backend."""
    
    def __init__(self):
        self.ui_path = Path(__file__).parent / "ui"
        self.backend_running = False
        self.ui_process = None
        
    async def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("🔍 Checking Prerequisites...")
        
        # Check Node.js
        try:
            result = subprocess.run(['node', '--version'], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info(f"✅ Node.js: {result.stdout.strip()}")
            else:
                logger.error("❌ Node.js not found")
                return False
        except Exception as e:
            logger.error(f"❌ Node.js check failed: {e}")
            return False
        
        # Check npm
        try:
            result = subprocess.run(['npm', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"✅ npm: {result.stdout.strip()}")
            else:
                logger.error("❌ npm not found")
                return False
        except Exception as e:
            logger.error(f"❌ npm check failed: {e}")
            return False
        
        # Check UI directory structure
        if not self.ui_path.exists():
            logger.error(f"❌ UI directory not found: {self.ui_path}")
            return False
        
        package_json = self.ui_path / "package.json"
        if not package_json.exists():
            logger.error("❌ package.json not found")
            return False
        
        logger.info("✅ All prerequisites met")
        return True
    
    async def install_dependencies(self) -> bool:
        """Install UI dependencies."""
        logger.info("📦 Installing UI Dependencies...")
        
        try:
            # Change to UI directory and install
            result = subprocess.run(
                ['npm', 'install'],
                cwd=self.ui_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Dependency installation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ Dependency installation timed out")
            return False
        except Exception as e:
            logger.error(f"❌ Dependency installation error: {e}")
            return False
    
    async def check_backend_services(self) -> bool:
        """Check if backend services are running."""
        logger.info("🔍 Checking Backend Services...")
        
        import aiohttp
        
        services = [
            ("FastAPI Backend", "http://localhost:8000/health"),
            ("ToT Reasoning Agent", "http://localhost:8001/health"),
            ("RAG Retrieval Agent", "http://localhost:8002/health")
        ]
        
        all_running = True
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in services:
                try:
                    async with session.get(url, timeout=5) as response:
                        if response.status == 200:
                            logger.info(f"✅ {service_name}: Running")
                        else:
                            logger.warning(f"⚠️ {service_name}: HTTP {response.status}")
                            all_running = False
                except Exception as e:
                    logger.warning(f"⚠️ {service_name}: Not accessible ({e})")
                    all_running = False
        
        if all_running:
            logger.info("✅ All backend services are running")
            self.backend_running = True
        else:
            logger.warning("⚠️ Some backend services are not running")
            logger.info("   UI will still work but with limited functionality")
        
        return True  # Don't fail if backend is not running
    
    async def test_ui_build(self) -> bool:
        """Test UI build process."""
        logger.info("🔨 Testing UI Build...")
        
        try:
            result = subprocess.run(
                ['npm', 'run', 'build'],
                cwd=self.ui_path,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutes timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ UI build successful")
                
                # Check if dist directory was created
                dist_path = self.ui_path / "dist"
                if dist_path.exists():
                    logger.info("✅ Build output directory created")
                    
                    # Check for essential files
                    index_html = dist_path / "index.html"
                    if index_html.exists():
                        logger.info("✅ index.html generated")
                    else:
                        logger.error("❌ index.html not found in build output")
                        return False
                    
                    return True
                else:
                    logger.error("❌ Build output directory not created")
                    return False
            else:
                logger.error(f"❌ UI build failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("❌ UI build timed out")
            return False
        except Exception as e:
            logger.error(f"❌ UI build error: {e}")
            return False
    
    async def start_ui_dev_server(self) -> bool:
        """Start UI development server."""
        logger.info("🚀 Starting UI Development Server...")
        
        try:
            # Start the dev server in background
            self.ui_process = subprocess.Popen(
                ['npm', 'run', 'dev'],
                cwd=self.ui_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Wait a moment for startup
            await asyncio.sleep(5)
            
            # Check if process is still running
            if self.ui_process.poll() is None:
                logger.info("✅ UI development server started")
                
                # Test if server is accessible
                import aiohttp
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get('http://localhost:3000', timeout=10) as response:
                            if response.status == 200:
                                logger.info("✅ UI server is accessible at http://localhost:3000")
                                return True
                            else:
                                logger.error(f"❌ UI server returned HTTP {response.status}")
                                return False
                except Exception as e:
                    logger.error(f"❌ UI server not accessible: {e}")
                    return False
            else:
                stdout, stderr = self.ui_process.communicate()
                logger.error(f"❌ UI server failed to start: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Failed to start UI server: {e}")
            return False
    
    async def test_ui_functionality(self) -> bool:
        """Test basic UI functionality."""
        logger.info("🧪 Testing UI Functionality...")
        
        # For now, just check if the server responds
        # In a full implementation, we could use Playwright for browser testing
        
        import aiohttp
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test main page
                async with session.get('http://localhost:3000', timeout=10) as response:
                    if response.status == 200:
                        content = await response.text()
                        if 'PyGent Factory' in content:
                            logger.info("✅ Main page loads correctly")
                        else:
                            logger.warning("⚠️ Main page content unexpected")
                    else:
                        logger.error(f"❌ Main page returned HTTP {response.status}")
                        return False
                
                # Test if static assets are served
                async with session.get('http://localhost:3000/vite.svg', timeout=5) as response:
                    if response.status == 200:
                        logger.info("✅ Static assets are served")
                    else:
                        logger.warning("⚠️ Static assets may not be served correctly")
                
                return True
                
        except Exception as e:
            logger.error(f"❌ UI functionality test failed: {e}")
            return False
    
    async def cleanup(self):
        """Clean up test resources."""
        if self.ui_process:
            logger.info("🧹 Stopping UI development server...")
            self.ui_process.terminate()
            try:
                self.ui_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.ui_process.kill()
            logger.info("✅ UI server stopped")
    
    async def run_complete_test(self) -> bool:
        """Run complete UI integration test."""
        logger.info("🚀 STARTING UI INTEGRATION TEST")
        logger.info("="*60)
        
        try:
            # Step 1: Check prerequisites
            if not await self.check_prerequisites():
                logger.error("❌ Prerequisites not met")
                return False
            
            # Step 2: Install dependencies
            if not await self.install_dependencies():
                logger.error("❌ Dependency installation failed")
                return False
            
            # Step 3: Check backend services
            await self.check_backend_services()
            
            # Step 4: Test build process
            if not await self.test_ui_build():
                logger.error("❌ UI build test failed")
                return False
            
            # Step 5: Start development server
            if not await self.start_ui_dev_server():
                logger.error("❌ UI server startup failed")
                return False
            
            # Step 6: Test functionality
            if not await self.test_ui_functionality():
                logger.error("❌ UI functionality test failed")
                return False
            
            # Success!
            logger.info("\n" + "="*60)
            logger.info("🎉 UI INTEGRATION TEST: SUCCESS!")
            logger.info("✅ Prerequisites met")
            logger.info("✅ Dependencies installed")
            logger.info("✅ Build process working")
            logger.info("✅ Development server running")
            logger.info("✅ Basic functionality verified")
            logger.info("")
            logger.info("🌐 UI accessible at: http://localhost:3000")
            
            if self.backend_running:
                logger.info("🔗 Backend services connected")
            else:
                logger.info("⚠️ Backend services not running (limited functionality)")
            
            logger.info("")
            logger.info("📋 Next Steps:")
            logger.info("1. Open http://localhost:3000 in your browser")
            logger.info("2. Test the multi-agent chat interface")
            logger.info("3. Verify real-time features work")
            logger.info("4. Check system monitoring dashboard")
            logger.info("5. Prepare for GitHub repository setup")
            logger.info("="*60)
            
            # Keep server running for manual testing
            logger.info("🔄 Keeping server running for manual testing...")
            logger.info("   Press Ctrl+C to stop")
            
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                logger.info("\n🛑 Stopping server...")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ UI integration test failed: {e}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            await self.cleanup()


async def main():
    """Run UI integration test."""
    tester = UIIntegrationTester()
    success = await tester.run_complete_test()
    return success


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)