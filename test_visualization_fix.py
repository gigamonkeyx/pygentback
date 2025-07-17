#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Visualization Fix
Quick test to verify the visualization generation works
"""

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_visualization():
    print('🎨 TESTING VISUALIZATION FIX')
    print('=' * 40)
    
    try:
        from sim.world_sim import WorldSimulation
        
        # Create and initialize simulation
        sim = WorldSimulation()
        init_success = await sim.initialize(num_agents=10)
        
        if init_success:
            # Run a quick simulation
            result = await sim.sim_loop(generations=2)
            behaviors = result['emergent_behaviors_detected']
            cooperation = result['cooperation_events']
            print(f'✅ Simulation: {behaviors} behaviors, {cooperation} cooperation')
            
            # Test visualization generation
            viz_success = sim.plot_emergence_evolution('test_visualization.png')
            
            if viz_success:
                print('🎉 VISUALIZATION: SUCCESS!')
                print('📊 Generated: test_visualization.png')
                
                # Check if file exists
                if os.path.exists('test_visualization.png'):
                    file_size = os.path.getsize('test_visualization.png')
                    print(f'📁 File size: {file_size} bytes')
                    print('✅ Visualization file created successfully!')
                else:
                    print('❌ File not found')
            else:
                print('❌ VISUALIZATION: FAILED')
        else:
            print('❌ Simulation initialization failed')
            
    except Exception as e:
        print(f'❌ Test failed: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_visualization())
