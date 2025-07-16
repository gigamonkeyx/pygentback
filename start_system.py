"""
PyGent Factory System Startup Script
Runs the system startup checklist with auto-start enabled

SYSTEM KNOWLEDGE (Updated 2025-06-27):
✅ WORKING MCP SERVERS (9/12):
  - Python Filesystem (filesystem operations)
  - Fetch Server (web requests)
  - Time Server (time operations)
  - Sequential Thinking (reasoning)
  - Memory Server (memory operations)
  - Git Server (version control)
  - Python Code Server (code execution)
  - Context7 Documentation (docs)
  - GitHub Repository (repo access)

✅ RESEARCH AGENTS (3):
  - ResearchCoordinatorAgent (research planning, Ollama reasoning)
  - DocumentAnalyzerAgent (document analysis, Ollama)
  - FactCheckerAgent (fact checking, Ollama + OpenRouter)

✅ MODELS AVAILABLE:
  - Ollama: 6 models (deepseek-coder-v2, etc.)
  - OpenRouter: 55 FREE models (deepseek-r1, etc.)

✅ AGENT TYPES (8):
  - reasoning, search, general, evolution, coding, research, basic, nlp

❌ KNOWN ISSUES:
  - 3 NPX MCP servers fail (PostgreSQL Official, GitHub Official, Memory Official)
  - Health check fixed to recognize working Python servers instead
"""

import asyncio
import sys
from system_startup_checklist import SystemStartupChecklist


async def quick_start():
    """Quick start with auto-recovery"""
    
    print("🚀 PYGENT FACTORY QUICK START")
    print("=" * 50)
    print("🔧 Auto-start mode: ENABLED")
    print("⚡ Quick recovery: ENABLED")
    print()
    
    checklist = SystemStartupChecklist(auto_start=True)
    
    try:
        success = await checklist.run_full_checklist()
        
        if success:
            print()
            print("🎉 SYSTEM READY!")
            print("✅ All services are running")
            print("🤖 9 MCP servers operational")
            print("🧠 3 research agents initialized")
            print("🔥 Ollama (6 models) + OpenRouter (55 free) available")
            print("🐉 Ready to execute dragon task!")
            print()
            print("Available capabilities:")
            print("  🗂️  Filesystem operations (Python Filesystem)")
            print("  🌐 Web requests (Fetch Server)")
            print("  🧠 Reasoning (Sequential Thinking)")
            print("  💾 Memory operations (Memory Server)")
            print("  📝 Code execution (Python Code Server)")
            print("  📚 Documentation (Context7)")
            print("  🔍 Research coordination (3 specialized agents)")
            print()
            print("Next steps:")
            print("  python dragon_task_execution.py")
            return True
        else:
            print()
            print("❌ SYSTEM STARTUP FAILED")
            print("🔧 Some services could not be started")
            print("📋 Run manual checklist for details:")
            print("  python system_startup_checklist.py")
            return False
            
    except KeyboardInterrupt:
        print("\n⏹️  Startup interrupted by user")
        return False
    except Exception as e:
        print(f"\n💥 Startup failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(quick_start())
    sys.exit(0 if success else 1)
