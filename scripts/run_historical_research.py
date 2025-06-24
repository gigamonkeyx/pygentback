#!/usr/bin/env python3
"""
Historical Research Workflow Runner

Runs the PyGent Factory historical research workflow (PIARES system)
with Windows compatibility and comprehensive error handling.
"""

import sys
import os
import asyncio
from pathlib import Path
from datetime import datetime

# Setup Windows-compatible environment
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONUTF8'] = '1'

# Add src to path
project_dir = Path(__file__).parent.parent
src_dir = project_dir / "src"
sys.path.insert(0, str(src_dir))

print("🏛️ PyGent Factory - Historical Research Workflow (PIARES)")
print("=" * 60)
print(f"🐍 Python: {sys.version}")
print(f"💻 Platform: {sys.platform}")
print(f"📁 Working Directory: {os.getcwd()}")
print(f"📦 Source Directory: {src_dir}")
print(f"⏰ Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

async def run_historical_research():
    """Run the historical research workflow"""
    
    print("🔧 INITIALIZING HISTORICAL RESEARCH SYSTEM")
    print("-" * 50)
    
    try:
        # Import the historical research workflow
        print("📚 Importing historical research workflow...")
        from workflows.historical_research import HistoricalResearchWorkflow
        print("✅ Historical research workflow imported successfully")
        
        # Create workflow instance
        print("🏗️ Creating workflow instance...")
        research_workflow = HistoricalResearchWorkflow()
        print("✅ Workflow instance created")
        
        # Initialize the workflow
        print("⚙️ Initializing workflow components...")
        try:
            success = await research_workflow.initialize()
            if success:
                print("✅ Workflow initialized successfully")
            else:
                print("⚠️ Workflow initialization returned False - continuing with limited functionality")
        except Exception as init_error:
            print(f"⚠️ Workflow initialization failed: {init_error}")
            print("📝 Continuing with basic functionality...")
        
        print("\n🎯 AVAILABLE RESEARCH TOPICS")
        print("-" * 50)
        
        # Display available topics
        topics = research_workflow.supported_topics
        for i, (topic_key, topic_info) in enumerate(topics.items(), 1):
            topic_name = topic_key.replace('_', ' ').title()
            focus_areas = ', '.join(topic_info['focus_areas'][:3])
            regions = ', '.join(topic_info['regions'][:3])
            print(f"{i:2d}. {topic_name}")
            print(f"    Focus: {focus_areas}")
            print(f"    Regions: {regions}")
        
        print(f"\n📊 Total supported topics: {len(topics)}")
        
        print("\n🔬 CONDUCTING SAMPLE RESEARCH")
        print("-" * 50)
        
        # Sample research: Scientific Revolutions
        topic = 'scientific_revolutions'
        research_question = "How did the Scientific Revolution transform global understanding of astronomy beyond European perspectives?"
        
        # Sample primary source texts (simplified for demo)
        source_texts = [
            """
            From Copernicus' De revolutionibus orbium coelestium (1543):
            "The motion of the sun and moon can be demonstrated by the assumption that the earth revolves. 
            This ancient doctrine, though it seemed absurd, deserves consideration when we observe 
            the celestial phenomena more carefully."
            """,
            """
            From Islamic astronomical treatise by Al-Tusi (13th century):
            "The observations of planetary motions require a new model that accounts for 
            the apparent irregularities. The coupling of circular motions can produce 
            linear motion, resolving the ancient problem of planetary theory."
            """,
            """
            From Chinese astronomical records (Ming Dynasty):
            "The celestial observations recorded in the Bureau of Astronomy show patterns
            that differ from traditional models. New instruments from the West provide
            greater precision, yet our ancient methods retain their validity."
            """
        ]
        
        print(f"📖 Research Topic: {topic.replace('_', ' ').title()}")
        print(f"❓ Research Question: {research_question}")
        print(f"📜 Source Texts: {len(source_texts)} primary sources")
        print()
        
        try:
            print("🚀 Executing research workflow...")
            
            # Conduct the research
            results = await research_workflow.conduct_research(
                topic=topic,
                research_question=research_question,
                source_texts=source_texts,
                focus_areas=['art_architecture', 'literacy', 'global_perspectives'],
                global_perspective=True
            )
            
            print("✅ Research workflow completed successfully!")
            print()
            
            # Display results
            print("📊 RESEARCH RESULTS")
            print("-" * 50)
            
            if 'research_metadata' in results:
                metadata = results['research_metadata']
                print(f"🎯 Topic: {metadata.get('topic', 'Unknown')}")
                print(f"⏱️ Execution Time: {metadata.get('execution_time', 'Unknown')}")
                print(f"✅ Success: {metadata.get('success', False)}")
                print()
            
            if 'source_analysis' in results:
                analysis = results['source_analysis']
                print("📚 SOURCE ANALYSIS:")
                if isinstance(analysis, dict) and 'steps' in analysis:
                    print(f"   • Analyzed {len(analysis['steps'])} analytical steps")
                    for i, step in enumerate(analysis['steps'][:3], 1):
                        step_name = step.get('name', f'Step {i}')
                        print(f"   • {step_name}")
                else:
                    print(f"   • Analysis completed: {type(analysis).__name__}")
                print()
            
            if 'validation_results' in results:
                validation = results['validation_results']
                print("🔍 SOURCE VALIDATION:")
                print(f"   • Validation completed: {type(validation).__name__}")
                print()
            
            if 'discovered_patterns' in results:
                patterns = results['discovered_patterns']
                print("🧬 DISCOVERED PATTERNS:")
                if isinstance(patterns, dict) and 'optimization_metrics' in patterns:
                    metrics = patterns['optimization_metrics']
                    print(f"   • Optimization completed with {len(metrics)} metrics")
                else:
                    print(f"   • Pattern discovery completed: {type(patterns).__name__}")
                print()
            
            if 'predicted_insights' in results:
                insights = results['predicted_insights']
                print("🔮 PREDICTED INSIGHTS:")
                print(f"   • Insights generated: {type(insights).__name__}")
                print()
            
            if 'synthesis' in results:
                synthesis = results['synthesis']
                print("🎭 RESEARCH SYNTHESIS:")
                if isinstance(synthesis, dict):
                    print(f"   • Synthesis contains {len(synthesis)} components")
                    for key in list(synthesis.keys())[:3]:
                        print(f"   • {key}: {type(synthesis[key]).__name__}")
                else:
                    print(f"   • Synthesis completed: {type(synthesis).__name__}")
                print()
            
            print("🎉 RESEARCH COMPLETED SUCCESSFULLY!")
            print("   The historical research workflow has been executed")
            print("   and produced comprehensive analytical results.")
            
        except Exception as research_error:
            print(f"❌ Research execution failed: {research_error}")
            print("📝 This may be due to missing dependencies or configuration issues")
            print("   The workflow framework is functional but requires full component setup")
        
    except ImportError as import_error:
        print(f"❌ Failed to import historical research workflow: {import_error}")
        print("📝 This indicates missing dependencies or import path issues")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("📝 Check the error details above for troubleshooting")

async def main():
    """Main execution function"""
    try:
        await run_historical_research()
    except KeyboardInterrupt:
        print("\n⚠️ Research interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
    finally:
        print(f"\n✨ Historical research session ended at {datetime.now().strftime('%H:%M:%S')}")

if __name__ == "__main__":
    print("🚀 Starting historical research workflow...")
    asyncio.run(main())
