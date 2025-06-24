#!/usr/bin/env python3
"""
Manual Documentation Build Trigger Script

Provides command-line interface for triggering documentation builds
with various options and configurations.
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from orchestration.intelligent_docs_builder import IntelligentDocsBuilder
from orchestration.build_trigger_detector import BuildTriggerDetector
from orchestration.mermaid_cache_manager import MermaidCacheManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_banner():
    """Print script banner"""
    print("🚀 PyGent Factory Documentation Build Trigger")
    print("=" * 50)

async def check_triggers():
    """Check current trigger status"""
    print("🔍 Checking build triggers...")
    
    detector = BuildTriggerDetector()
    status = await detector.check_all_triggers()
    
    print(f"\n📊 Trigger Status:")
    print(f"  Should Build: {'✅ Yes' if status['should_build'] else '❌ No'}")
    
    if status['should_build']:
        print(f"  Trigger Reasons:")
        for reason in status['summary']['trigger_reasons']:
            print(f"    • {reason}")
    
    # Show detailed trigger information
    triggers = status['triggers']
    
    if triggers['git']['new_commits']:
        print(f"\n📝 Git Changes:")
        print(f"  New Commits: {len(triggers['git']['commits'])}")
        for commit in triggers['git']['commits'][:3]:  # Show first 3
            print(f"    • {commit['hash'][:8]}: {commit['message']}")
    
    if triggers['files']['files_changed']:
        print(f"\n📁 File Changes:")
        details = triggers['files']['details']
        print(f"  Changed: {details['changed_count']}")
        print(f"  New: {details['new_count']}")
        print(f"  Deleted: {details['deleted_count']}")
    
    return status

async def check_mermaid_cache():
    """Check Mermaid diagram cache status"""
    print("🎨 Checking Mermaid diagram cache...")
    
    manager = MermaidCacheManager()
    status = manager.get_cache_status()
    
    print(f"\n📊 Mermaid Cache Status:")
    print(f"  Total Diagrams: {status['total_diagrams']}")
    print(f"  Cached: {status['cached_diagrams']}")
    print(f"  Missing: {status['missing_diagrams']}")
    print(f"  Outdated: {status['outdated_diagrams']}")
    print(f"  Cache Size: {status['cache_size_mb']:.1f} MB")
    
    if status['last_updated']:
        print(f"  Last Updated: {status['last_updated']}")
    
    return status

async def regenerate_mermaid(force=False):
    """Regenerate Mermaid diagrams"""
    print(f"🎨 Regenerating Mermaid diagrams (force={force})...")
    
    manager = MermaidCacheManager()
    results = await manager.regenerate_diagrams(force=force)
    
    if results:
        successful = sum(1 for success in results.values() if success)
        print(f"\n✅ Mermaid regeneration complete: {successful}/{len(results)} successful")
        
        for diagram_id, success in results.items():
            status = "✅" if success else "❌"
            print(f"  {status} {diagram_id}")
    else:
        print("\n✅ All Mermaid diagrams are up to date")

async def execute_build(force=False, production=True, mermaid_only=False):
    """Execute documentation build"""
    if mermaid_only:
        await regenerate_mermaid(force=force)
        return
    
    print(f"🏗️  Executing documentation build (force={force}, production={production})...")
    
    builder = IntelligentDocsBuilder()
    result = await builder.execute_intelligent_build(
        force=force,
        production=production
    )
    
    print(f"\n📊 Build Results:")
    print(f"  Build ID: {result['build_id']}")
    print(f"  Success: {'✅ Yes' if result['success'] else '❌ No'}")
    print(f"  Duration: {result['total_duration_seconds']:.1f}s")
    
    if result.get('skipped'):
        print("  Status: ⏭️  Skipped (no triggers)")
        return result
    
    # Show trigger information
    if 'triggers' in result and not result.get('force'):
        triggers = result['triggers']
        if triggers.get('summary', {}).get('trigger_reasons'):
            print(f"  Triggered by: {', '.join(triggers['summary']['trigger_reasons'])}")
    
    # Show Mermaid results
    if result.get('mermaid_results'):
        mermaid_count = len(result['mermaid_results'])
        mermaid_success = sum(1 for success in result['mermaid_results'].values() if success)
        print(f"  Mermaid Diagrams: {mermaid_success}/{mermaid_count} regenerated")
    
    # Show build results
    if 'build_results' in result:
        br = result['build_results']
        if br['success']:
            print(f"  Files Generated: {br.get('files_generated', 0)}")
            print(f"  Output Size: {br.get('output_size_mb', 0):.1f} MB")
        else:
            print(f"  Build Error: {br.get('error', 'Unknown error')}")
    
    if result.get('error'):
        print(f"  Error: {result['error']}")
    
    return result

async def get_build_status():
    """Get comprehensive build system status"""
    print("📊 Getting build system status...")
    
    builder = IntelligentDocsBuilder()
    status = await builder.get_build_status()
    
    print(f"\n📊 Build System Status:")
    print(f"  System: {status['system_status']}")
    
    # Trigger status
    triggers = status['triggers']
    print(f"  Should Build: {'✅ Yes' if triggers['should_build'] else '❌ No'}")
    
    # Mermaid cache
    cache = status['mermaid_cache']
    print(f"  Mermaid Cache: {cache['cached_diagrams']}/{cache['total_diagrams']} cached")
    
    # Build history
    history = status['build_history']
    print(f"  Build History: {history['total_builds']} total, {history['success_rate']:.1%} success rate")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    for rec in status['recommendations']:
        print(f"  • {rec}")

def create_trigger_file():
    """Create manual trigger file"""
    trigger_file = Path("src/docs/.force_rebuild")
    trigger_file.touch()
    print(f"✅ Created trigger file: {trigger_file}")
    print("   Next documentation build will be forced")

async def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="PyGent Factory Documentation Build Trigger",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s status                    # Check build system status
  %(prog)s check                     # Check trigger conditions
  %(prog)s build                     # Build if triggers detected
  %(prog)s build --force             # Force build regardless of triggers
  %(prog)s build --dev               # Development build
  %(prog)s mermaid                   # Check Mermaid cache status
  %(prog)s mermaid --regenerate      # Regenerate outdated diagrams
  %(prog)s mermaid --force           # Force regenerate all diagrams
  %(prog)s trigger                   # Create manual trigger file
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Status command
    subparsers.add_parser('status', help='Get build system status')
    
    # Check command
    subparsers.add_parser('check', help='Check trigger conditions')
    
    # Build command
    build_parser = subparsers.add_parser('build', help='Execute documentation build')
    build_parser.add_argument('--force', action='store_true', help='Force build regardless of triggers')
    build_parser.add_argument('--dev', action='store_true', help='Development build (faster)')
    build_parser.add_argument('--mermaid-only', action='store_true', help='Only regenerate Mermaid diagrams')
    
    # Mermaid command
    mermaid_parser = subparsers.add_parser('mermaid', help='Mermaid diagram operations')
    mermaid_parser.add_argument('--regenerate', action='store_true', help='Regenerate outdated diagrams')
    mermaid_parser.add_argument('--force', action='store_true', help='Force regenerate all diagrams')
    
    # Trigger command
    subparsers.add_parser('trigger', help='Create manual trigger file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    print_banner()
    
    try:
        if args.command == 'status':
            await get_build_status()
        
        elif args.command == 'check':
            await check_triggers()
        
        elif args.command == 'build':
            result = await execute_build(
                force=args.force,
                production=not args.dev,
                mermaid_only=args.mermaid_only
            )
            if not result['success'] and not result.get('skipped'):
                sys.exit(1)
        
        elif args.command == 'mermaid':
            if args.regenerate or args.force:
                await regenerate_mermaid(force=args.force)
            else:
                await check_mermaid_cache()
        
        elif args.command == 'trigger':
            create_trigger_file()
        
        print("\n✅ Operation completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        logger.exception("Operation failed")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
