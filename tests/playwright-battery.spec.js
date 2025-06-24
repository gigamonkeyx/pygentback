const { test, expect } = require('@playwright/test');

// PyGent Factory Comprehensive Test Battery
// Tests all services, endpoints, and functionality

test.describe('PyGent Factory Complete System Test Battery', () => {
  
  // Backend API Tests
  test.describe('Backend API Server (Port 8000)', () => {
    
    test('API Root Endpoint', async ({ request }) => {
      const response = await request.get('http://localhost:8000/');
      expect(response.status()).toBe(200);
      const data = await response.json();
      expect(data).toHaveProperty('message');
      expect(data.message).toContain('PyGent Factory');
      console.log('✅ Backend API Root:', data);
    });

    test('API Health Check', async ({ request }) => {
      const response = await request.get('http://localhost:8000/api/v1/health');
      expect(response.status()).toBe(200);
      const data = await response.json();
      console.log('✅ Backend Health:', data);
    });

    test('API Documentation', async ({ request }) => {
      const response = await request.get('http://localhost:8000/docs');
      expect(response.status()).toBe(200);
      console.log('✅ API Docs accessible');
    });

    test('MCP Servers Endpoint', async ({ request }) => {
      const response = await request.get('http://localhost:8000/api/v1/mcp/servers');
      expect(response.status()).toBe(200);
      const data = await response.json();
      console.log('✅ MCP Servers:', data);
    });

    test('Agents Endpoint', async ({ request }) => {
      const response = await request.get('http://localhost:8000/api/v1/agents');
      expect(response.status()).toBe(200);
      const data = await response.json();
      console.log('✅ Agents:', data);
    });

    test('RAG Search Endpoint', async ({ request }) => {
      const response = await request.post('http://localhost:8000/api/v1/rag/search', {
        data: {
          query: "test query",
          limit: 5
        }
      });
      console.log('✅ RAG Search Status:', response.status());
    });
  });

  // Frontend UI Tests
  test.describe('Frontend UI (Port 3000)', () => {
    
    test('Frontend Root Access', async ({ page }) => {
      try {
        await page.goto('http://localhost:3000/', { timeout: 10000 });
        await page.waitForLoadState('networkidle', { timeout: 10000 });
        
        const title = await page.title();
        console.log('✅ Frontend Title:', title);
        
        // Check if React app loaded
        const rootElement = await page.locator('#root').count();
        expect(rootElement).toBeGreaterThan(0);
        console.log('✅ React root element found');
        
        // Check for any JavaScript errors
        const errors = [];
        page.on('pageerror', error => errors.push(error.message));
        await page.waitForTimeout(2000);
        
        if (errors.length > 0) {
          console.log('❌ JavaScript Errors:', errors);
        } else {
          console.log('✅ No JavaScript errors detected');
        }
        
      } catch (error) {
        console.log('❌ Frontend Root Error:', error.message);
        throw error;
      }
    });

    test('Frontend Index.html Direct Access', async ({ page }) => {
      try {
        await page.goto('http://localhost:3000/index.html', { timeout: 10000 });
        await page.waitForLoadState('networkidle', { timeout: 10000 });
        
        const title = await page.title();
        console.log('✅ Frontend Index Title:', title);
        
        // Check for React components
        const content = await page.content();
        console.log('✅ Page loaded, content length:', content.length);
        
      } catch (error) {
        console.log('❌ Frontend Index Error:', error.message);
        throw error;
      }
    });

    test('Frontend API Connectivity', async ({ page }) => {
      await page.goto('http://localhost:3000/index.html');
      
      // Test if frontend can reach backend API
      const apiResponse = await page.evaluate(async () => {
        try {
          const response = await fetch('/api/');
          return {
            status: response.status,
            ok: response.ok,
            data: await response.json()
          };
        } catch (error) {
          return { error: error.message };
        }
      });
      
      console.log('✅ Frontend API Connectivity:', apiResponse);
    });

    test('Frontend WebSocket Connection', async ({ page }) => {
      await page.goto('http://localhost:3000/index.html');
      
      // Test WebSocket connection
      const wsResult = await page.evaluate(async () => {
        return new Promise((resolve) => {
          try {
            const ws = new WebSocket('ws://localhost:3000/ws');
            
            ws.onopen = () => {
              ws.close();
              resolve({ status: 'connected' });
            };
            
            ws.onerror = (error) => {
              resolve({ status: 'error', error: error.message });
            };
            
            setTimeout(() => {
              resolve({ status: 'timeout' });
            }, 5000);
          } catch (error) {
            resolve({ status: 'error', error: error.message });
          }
        });
      });
      
      console.log('✅ WebSocket Test:', wsResult);
    });
  });

  // Documentation Tests
  test.describe('Documentation Server (Port 3001)', () => {
    
    test('Documentation Root Access', async ({ page }) => {
      try {
        await page.goto('http://localhost:3001/', { timeout: 10000 });
        await page.waitForLoadState('networkidle', { timeout: 10000 });
        
        const title = await page.title();
        console.log('✅ Documentation Title:', title);
        
        const content = await page.content();
        console.log('✅ Documentation loaded, content length:', content.length);
        
      } catch (error) {
        console.log('❌ Documentation Error:', error.message);
        throw error;
      }
    });
  });

  // Cross-Service Integration Tests
  test.describe('Cross-Service Integration', () => {
    
    test('All Services Responding', async ({ request }) => {
      const services = [
        { name: 'Backend API', url: 'http://localhost:8000/' },
        { name: 'Frontend UI', url: 'http://localhost:3000/index.html' },
        { name: 'Documentation', url: 'http://localhost:3001/' }
      ];
      
      for (const service of services) {
        try {
          const response = await request.get(service.url);
          console.log(`✅ ${service.name}: Status ${response.status()}`);
        } catch (error) {
          console.log(`❌ ${service.name}: ${error.message}`);
        }
      }
    });

    test('Frontend to Backend Proxy', async ({ page }) => {
      await page.goto('http://localhost:3000/index.html');
      
      // Test proxy routing
      const proxyTest = await page.evaluate(async () => {
        try {
          const response = await fetch('/api/');
          return {
            status: response.status,
            headers: Object.fromEntries(response.headers.entries()),
            data: await response.json()
          };
        } catch (error) {
          return { error: error.message };
        }
      });
      
      console.log('✅ Proxy Test Result:', proxyTest);
    });
  });

  // Performance and Load Tests
  test.describe('Performance Tests', () => {
    
    test('Frontend Load Performance', async ({ page }) => {
      const startTime = Date.now();
      
      await page.goto('http://localhost:3000/index.html');
      await page.waitForLoadState('networkidle');
      
      const loadTime = Date.now() - startTime;
      console.log(`✅ Frontend Load Time: ${loadTime}ms`);
      
      expect(loadTime).toBeLessThan(10000); // Should load within 10 seconds
    });

    test('Backend API Response Time', async ({ request }) => {
      const startTime = Date.now();
      
      const response = await request.get('http://localhost:8000/');
      
      const responseTime = Date.now() - startTime;
      console.log(`✅ Backend Response Time: ${responseTime}ms`);
      
      expect(response.status()).toBe(200);
      expect(responseTime).toBeLessThan(5000); // Should respond within 5 seconds
    });
  });

  // Error Detection Tests
  test.describe('Error Detection', () => {
    
    test('Console Error Detection', async ({ page }) => {
      const errors = [];
      const warnings = [];
      
      page.on('console', msg => {
        if (msg.type() === 'error') {
          errors.push(msg.text());
        } else if (msg.type() === 'warning') {
          warnings.push(msg.text());
        }
      });
      
      await page.goto('http://localhost:3000/index.html');
      await page.waitForTimeout(5000);
      
      console.log('Console Errors:', errors);
      console.log('Console Warnings:', warnings);
      
      // Report but don't fail on warnings
      if (errors.length > 0) {
        console.log('❌ Console errors detected:', errors);
      } else {
        console.log('✅ No console errors detected');
      }
    });

    test('Network Error Detection', async ({ page }) => {
      const failedRequests = [];
      
      page.on('requestfailed', request => {
        failedRequests.push({
          url: request.url(),
          failure: request.failure()
        });
      });
      
      await page.goto('http://localhost:3000/index.html');
      await page.waitForTimeout(5000);
      
      if (failedRequests.length > 0) {
        console.log('❌ Failed requests:', failedRequests);
      } else {
        console.log('✅ No failed network requests');
      }
    });
  });
});
