const { test, expect } = require('@playwright/test');

// Quick Diagnostic Test - Fast service health check
test.describe('PyGent Factory Quick Diagnostic', () => {
  
  test('Quick Service Health Check', async ({ request, page }) => {
    console.log('🔍 Starting Quick Diagnostic...');
    
    const results = {
      backend: { status: 'unknown', details: null },
      frontend: { status: 'unknown', details: null },
      documentation: { status: 'unknown', details: null },
      integration: { status: 'unknown', details: null }
    };
    
    // Test Backend API
    try {
      console.log('Testing Backend API...');
      const backendResponse = await request.get('http://localhost:8000/', { timeout: 5000 });
      results.backend.status = backendResponse.status() === 200 ? 'healthy' : 'unhealthy';
      results.backend.details = {
        status: backendResponse.status(),
        data: await backendResponse.json().catch(() => null)
      };
      console.log(`✅ Backend API: ${results.backend.status} (${backendResponse.status()})`);
    } catch (error) {
      results.backend.status = 'error';
      results.backend.details = error.message;
      console.log(`❌ Backend API: ${error.message}`);
    }
    
    // Test Frontend UI
    try {
      console.log('Testing Frontend UI...');
      await page.goto('http://localhost:3000/index.html', { timeout: 10000 });
      await page.waitForLoadState('domcontentloaded', { timeout: 5000 });
      
      const title = await page.title();
      const hasRoot = await page.locator('#root').count() > 0;
      
      results.frontend.status = hasRoot ? 'healthy' : 'partial';
      results.frontend.details = { title, hasRoot };
      console.log(`✅ Frontend UI: ${results.frontend.status} (${title})`);
    } catch (error) {
      results.frontend.status = 'error';
      results.frontend.details = error.message;
      console.log(`❌ Frontend UI: ${error.message}`);
    }
    
    // Test Documentation
    try {
      console.log('Testing Documentation...');
      const docsResponse = await request.get('http://localhost:3001/', { timeout: 5000 });
      results.documentation.status = docsResponse.status() === 200 ? 'healthy' : 'unhealthy';
      results.documentation.details = { status: docsResponse.status() };
      console.log(`✅ Documentation: ${results.documentation.status} (${docsResponse.status()})`);
    } catch (error) {
      results.documentation.status = 'error';
      results.documentation.details = error.message;
      console.log(`❌ Documentation: ${error.message}`);
    }
    
    // Test Frontend-Backend Integration
    if (results.frontend.status === 'healthy') {
      try {
        console.log('Testing Frontend-Backend Integration...');
        await page.goto('http://localhost:3000/index.html');
        
        const apiTest = await page.evaluate(async () => {
          try {
            const response = await fetch('/api/', { timeout: 5000 });
            return { status: response.status, ok: response.ok };
          } catch (error) {
            return { error: error.message };
          }
        });
        
        results.integration.status = apiTest.ok ? 'healthy' : 'partial';
        results.integration.details = apiTest;
        console.log(`✅ Integration: ${results.integration.status}`);
      } catch (error) {
        results.integration.status = 'error';
        results.integration.details = error.message;
        console.log(`❌ Integration: ${error.message}`);
      }
    }
    
    // Summary Report
    console.log('\n📊 QUICK DIAGNOSTIC SUMMARY:');
    console.log('================================');
    console.log(`Backend API:     ${results.backend.status.toUpperCase()}`);
    console.log(`Frontend UI:     ${results.frontend.status.toUpperCase()}`);
    console.log(`Documentation:   ${results.documentation.status.toUpperCase()}`);
    console.log(`Integration:     ${results.integration.status.toUpperCase()}`);
    
    // Detailed Results
    console.log('\n🔍 DETAILED RESULTS:');
    console.log(JSON.stringify(results, null, 2));
    
    // Determine overall health
    const healthyServices = Object.values(results).filter(r => r.status === 'healthy').length;
    const totalServices = Object.keys(results).length;
    
    console.log(`\n🎯 OVERALL HEALTH: ${healthyServices}/${totalServices} services healthy`);
    
    if (healthyServices === totalServices) {
      console.log('✅ All services are healthy - ready for full test battery');
    } else if (healthyServices >= totalServices / 2) {
      console.log('⚠️  Some services have issues - investigate before full testing');
    } else {
      console.log('❌ Major service issues detected - fix before proceeding');
    }
    
    // Store results for potential use by other tests
    global.diagnosticResults = results;
  });
  
  test('Service Connectivity Matrix', async ({ request }) => {
    console.log('\n🔗 Testing Service Connectivity Matrix...');
    
    const endpoints = [
      'http://localhost:8000/',
      'http://localhost:8000/api/v1/health',
      'http://localhost:8000/docs',
      'http://localhost:3000/index.html',
      'http://localhost:3001/'
    ];
    
    const matrix = {};
    
    for (const endpoint of endpoints) {
      try {
        const response = await request.get(endpoint, { timeout: 3000 });
        matrix[endpoint] = {
          status: response.status(),
          ok: response.ok(),
          contentType: response.headers()['content-type'] || 'unknown'
        };
        console.log(`✅ ${endpoint}: ${response.status()}`);
      } catch (error) {
        matrix[endpoint] = {
          status: 'error',
          error: error.message
        };
        console.log(`❌ ${endpoint}: ${error.message}`);
      }
    }
    
    console.log('\n📋 Connectivity Matrix:');
    console.log(JSON.stringify(matrix, null, 2));
  });
});
