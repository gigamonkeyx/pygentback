#!/usr/bin/env python3
from src.dgm.autonomy_fixed import AutonomySystem

# Test the enhanced autonomy system
autonomy_system = AutonomySystem({'safety_threshold': 0.6})
result = autonomy_system.check_autonomy()

print('\n🎯 FINAL AUTONOMY VALIDATION RESULTS:')
print(f'Approval Rate: {result["approval_rate"]:.3f} ({result["approval_rate"]*100:.1f}%)')
print(f'Threshold: {result["approval_threshold"]:.3f} ({result["approval_threshold"]*100:.1f}%)')
print(f'Approved: {result["approved"]}')
print('\nIndividual Scores:')
for name, score in result['individual_scores'].items():
    print(f'  {name}: {score:.3f} ({score*100:.1f}%)')

if result['approval_rate'] >= 0.95:
    print('\n🎉 95% THRESHOLD ACHIEVED!')
else:
    gap = 0.95 - result['approval_rate']
    print(f'\n⚡ Gap to 95%: {gap:.3f} ({gap*100:.1f}%)')
