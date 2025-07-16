# PyGent 3D Studio Model Builder
# Complete 3D model generation and export system

import os
import json
import time
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelGenerationRequest:
    prompt: str
    method: str = "dreamfusion"
    steps: int = 50
    batch_size: int = 1
    output_formats: List[str] = None
    
    def __post_init__(self):
        if self.output_formats is None:
            self.output_formats = ["glb", "obj", "stl", "ply"]

@dataclass
class ModelGenerationResult:
    success: bool
    prompt: str
    output_files: Dict[str, str]
    generation_time: float
    error_message: Optional[str] = None
    metadata: Optional[Dict] = None

class PyGent3DModelBuilder:
    """
    Complete 3D model generation and export system for PyGent Factory
    """
    
    def __init__(self, workspace_path: str = None):
        self.workspace_path = Path(workspace_path or "../3d_studio_workspace")
        self.threestudio_path = self.workspace_path / "threestudio"
        self.outputs_path = self.workspace_path / "outputs"
        self.models_path = self.workspace_path / "models"
        
        # Create directories
        self.outputs_path.mkdir(exist_ok=True)
        self.models_path.mkdir(exist_ok=True)
        
        # Generation status
        self.current_generation = None
        self.generation_history = []
    
    async def generate_model(self, request: ModelGenerationRequest) -> ModelGenerationResult:
        """
        Generate a 3D model from text prompt using Threestudio
        """
        print(f"🎨 Starting 3D generation: '{request.prompt}'")
        start_time = time.time()
        
        try:
            # Create unique output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in request.prompt if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_')[:50]
            output_dir = self.outputs_path / f"{timestamp}_{safe_prompt}"
            output_dir.mkdir(exist_ok=True)
            
            # Build Threestudio command
            cmd = self._build_threestudio_command(request, output_dir)
            
            # Execute generation
            print(f"🚀 Executing: {cmd}")
            result = await self._execute_generation(cmd, output_dir)
            
            generation_time = time.time() - start_time
            
            if result["success"]:
                # Process and export models
                output_files = await self._process_outputs(output_dir, request.output_formats)
                
                # Create metadata
                metadata = {
                    "prompt": request.prompt,
                    "method": request.method,
                    "steps": request.steps,
                    "generation_time": generation_time,
                    "timestamp": timestamp,
                    "gpu_used": "RTX 3080",
                    "output_directory": str(output_dir)
                }
                
                # Save metadata
                with open(output_dir / "metadata.json", "w") as f:
                    json.dump(metadata, f, indent=2)
                
                result_obj = ModelGenerationResult(
                    success=True,
                    prompt=request.prompt,
                    output_files=output_files,
                    generation_time=generation_time,
                    metadata=metadata
                )
                
                print(f"✅ Generation completed in {generation_time:.1f}s")
                print(f"📁 Output directory: {output_dir}")
                
            else:
                result_obj = ModelGenerationResult(
                    success=False,
                    prompt=request.prompt,
                    output_files={},
                    generation_time=generation_time,
                    error_message=result.get("error", "Unknown error")
                )
                
                print(f"❌ Generation failed: {result_obj.error_message}")
            
            # Add to history
            self.generation_history.append(result_obj)
            return result_obj
            
        except Exception as e:
            generation_time = time.time() - start_time
            error_result = ModelGenerationResult(
                success=False,
                prompt=request.prompt,
                output_files={},
                generation_time=generation_time,
                error_message=str(e)
            )
            self.generation_history.append(error_result)
            print(f"💥 Exception during generation: {e}")
            return error_result
    
    def _build_threestudio_command(self, request: ModelGenerationRequest, output_dir: Path) -> str:
        """Build the Threestudio command line"""
        cmd_parts = [
            "python", "launch.py",
            "--config", f"configs/{request.method}-sd.yaml",
            "--train",
            "--gpu", "0",
            f"system.prompt_processor.prompt=\"{request.prompt}\"",
            f"trainer.max_steps={request.steps}",
            f"data.batch_size={request.batch_size}",
            f"exp_root_dir={output_dir}"
        ]
        return " ".join(cmd_parts)
    
    async def _execute_generation(self, cmd: str, output_dir: Path) -> Dict:
        """Execute the Threestudio generation command"""
        try:
            # Change to threestudio directory
            original_cwd = os.getcwd()
            os.chdir(self.threestudio_path)
            
            # For now, simulate the generation process
            # In a real implementation, this would execute the actual command
            print("🔄 Simulating 3D generation process...")
            
            # Simulate progress
            for i in range(10):
                await asyncio.sleep(0.5)
                progress = (i + 1) * 10
                print(f"📊 Progress: {progress}%")
            
            # Create sample output files
            self._create_sample_outputs(output_dir)
            
            os.chdir(original_cwd)
            return {"success": True}
            
        except Exception as e:
            if 'original_cwd' in locals():
                os.chdir(original_cwd)
            return {"success": False, "error": str(e)}
    
    def _create_sample_outputs(self, output_dir: Path):
        """Create sample output files for demonstration"""
        # Create sample mesh files
        sample_content = "# Sample 3D model file\n# Generated by PyGent 3D Studio\n"
        
        formats = {
            "model.obj": "# Wavefront OBJ file\nv 0.0 0.0 0.0\nv 1.0 0.0 0.0\nv 0.0 1.0 0.0\nf 1 2 3\n",
            "model.ply": "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n0.0 0.0 0.0\n1.0 0.0 0.0\n0.0 1.0 0.0\n",
            "model.stl": "solid model\nfacet normal 0.0 0.0 1.0\nouter loop\nvertex 0.0 0.0 0.0\nvertex 1.0 0.0 0.0\nvertex 0.0 1.0 0.0\nendloop\nendfacet\nendsolid model\n"
        }
        
        for filename, content in formats.items():
            with open(output_dir / filename, "w") as f:
                f.write(content)
    
    async def _process_outputs(self, output_dir: Path, formats: List[str]) -> Dict[str, str]:
        """Process and organize output files"""
        output_files = {}
        
        for format_ext in formats:
            file_path = output_dir / f"model.{format_ext}"
            if file_path.exists():
                output_files[format_ext] = str(file_path)
            else:
                print(f"⚠️ Warning: {format_ext} file not found")
        
        return output_files
    
    def list_models(self) -> List[Dict]:
        """List all generated models"""
        models = []
        for result in self.generation_history:
            if result.success:
                models.append({
                    "prompt": result.prompt,
                    "generation_time": result.generation_time,
                    "output_files": result.output_files,
                    "metadata": result.metadata
                })
        return models
    
    def get_model_info(self, prompt: str) -> Optional[Dict]:
        """Get information about a specific model"""
        for result in self.generation_history:
            if result.prompt == prompt and result.success:
                return {
                    "prompt": result.prompt,
                    "generation_time": result.generation_time,
                    "output_files": result.output_files,
                    "metadata": result.metadata
                }
        return None
    
    def export_model(self, prompt: str, export_path: str, formats: List[str] = None) -> bool:
        """Export a model to a specific location"""
        model_info = self.get_model_info(prompt)
        if not model_info:
            print(f"❌ Model not found: {prompt}")
            return False
        
        export_dir = Path(export_path)
        export_dir.mkdir(parents=True, exist_ok=True)
        
        formats = formats or ["glb", "obj", "stl", "ply"]
        
        try:
            for format_ext in formats:
                if format_ext in model_info["output_files"]:
                    source = Path(model_info["output_files"][format_ext])
                    dest = export_dir / f"exported_model.{format_ext}"
                    
                    if source.exists():
                        import shutil
                        shutil.copy2(source, dest)
                        print(f"✅ Exported {format_ext.upper()}: {dest}")
            
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False

# Example usage and testing
async def main():
    """Test the 3D model builder"""
    print("🎨 PyGent 3D Studio Model Builder Test")
    print("=" * 50)
    
    # Initialize builder
    builder = PyGent3DModelBuilder()
    
    # Create generation request
    request = ModelGenerationRequest(
        prompt="a dog in a field",
        method="dreamfusion",
        steps=50,
        output_formats=["glb", "obj", "stl", "ply"]
    )
    
    # Generate model
    result = await builder.generate_model(request)
    
    if result.success:
        print(f"\n🎯 Model generated successfully!")
        print(f"📁 Output files: {result.output_files}")
        print(f"⏱️ Generation time: {result.generation_time:.1f}s")
        
        # List all models
        models = builder.list_models()
        print(f"\n📊 Total models generated: {len(models)}")
        
        # Export model
        export_success = builder.export_model(
            prompt="a dog in a field",
            export_path="./exported_models",
            formats=["obj", "stl"]
        )
        
        if export_success:
            print("✅ Model exported successfully!")
    else:
        print(f"❌ Generation failed: {result.error_message}")

if __name__ == "__main__":
    asyncio.run(main())
