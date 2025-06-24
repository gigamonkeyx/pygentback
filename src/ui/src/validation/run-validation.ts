/**
 * Run Startup Service Integration Validation
 * Execute validation tests and display results
 */

import { validateStartupServiceIntegration, formatValidationResults } from './startup-integration-validation';

/**
 * Main validation runner
 */
async function runValidation() {
  console.log('🚀 Starting PyGent Factory Startup Service Integration Validation...\n');

  try {
    const results = await validateStartupServiceIntegration();
    const formattedResults = formatValidationResults(results);
    
    console.log(formattedResults);
    
    if (results.successRate === 100) {
      console.log('🎉 ALL TESTS PASSED! Integration is ready for Phase 2.');
    } else if (results.successRate >= 80) {
      console.log('⚠️  Most tests passed, but some issues need attention.');
    } else {
      console.log('❌ Multiple integration issues detected. Review required.');
    }

    return results;
  } catch (error) {
    console.error('💥 Validation failed with error:', error);
    throw error;
  }
}

// Export for use in other modules
export { runValidation };

// Run validation if this file is executed directly
if (require.main === module) {
  runValidation()
    .then((results) => {
      process.exit(results.successRate === 100 ? 0 : 1);
    })
    .catch((error) => {
      console.error('Validation execution failed:', error);
      process.exit(1);
    });
}
