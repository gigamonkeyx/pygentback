# PyGent Factory Comprehensive Test Battery Runner
# PowerShell script to run complete system diagnostics

Write-Host "🚀 PyGent Factory Test Battery Starting..." -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

# Check if we're in the right directory
if (!(Test-Path "playwright-battery.spec.js")) {
    Write-Host "❌ Error: Must run from tests directory" -ForegroundColor Red
    Write-Host "Current directory: $(Get-Location)" -ForegroundColor Yellow
    Write-Host "Expected files: playwright-battery.spec.js, package.json" -ForegroundColor Yellow
    exit 1
}

# Install dependencies if needed
Write-Host "📦 Checking Playwright installation..." -ForegroundColor Blue
if (!(Test-Path "node_modules")) {
    Write-Host "Installing Playwright dependencies..." -ForegroundColor Yellow
    npm install
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Failed to install dependencies" -ForegroundColor Red
        exit 1
    }
}

# Install browsers if needed
Write-Host "🌐 Installing Playwright browsers..." -ForegroundColor Blue
npx playwright install chromium
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install browsers" -ForegroundColor Red
    exit 1
}

# Pre-flight service checks
Write-Host "🔍 Pre-flight Service Checks..." -ForegroundColor Blue

$services = @(
    @{ Name = "Backend API"; URL = "http://localhost:8000/" },
    @{ Name = "Frontend UI"; URL = "http://localhost:3000/index.html" },
    @{ Name = "Documentation"; URL = "http://localhost:3001/" }
)

foreach ($service in $services) {
    try {
        Write-Host "Checking $($service.Name)..." -ForegroundColor Cyan
        $response = Invoke-WebRequest -Uri $service.URL -TimeoutSec 10 -UseBasicParsing
        Write-Host "✅ $($service.Name): Status $($response.StatusCode)" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $($service.Name): $($_.Exception.Message)" -ForegroundColor Red
    }
}

# Run the comprehensive test battery
Write-Host "🧪 Running Playwright Test Battery..." -ForegroundColor Blue
Write-Host "This will test all services, endpoints, and integrations" -ForegroundColor Yellow

# Run tests with detailed output
npx playwright test --reporter=list,html,json

$testExitCode = $LASTEXITCODE

# Generate summary report
Write-Host "📊 Test Battery Complete!" -ForegroundColor Green
Write-Host "=============================================" -ForegroundColor Green

if ($testExitCode -eq 0) {
    Write-Host "✅ All tests passed successfully!" -ForegroundColor Green
} else {
    Write-Host "❌ Some tests failed. Check the detailed report." -ForegroundColor Red
}

# Show report locations
Write-Host "📋 Reports generated:" -ForegroundColor Blue
Write-Host "  - HTML Report: playwright-report/index.html" -ForegroundColor Cyan
Write-Host "  - JSON Report: test-results.json" -ForegroundColor Cyan

# Offer to open HTML report
$openReport = Read-Host "Open HTML report in browser? (y/n)"
if ($openReport -eq "y" -or $openReport -eq "Y") {
    if (Test-Path "playwright-report/index.html") {
        Start-Process "playwright-report/index.html"
    } else {
        Write-Host "❌ HTML report not found" -ForegroundColor Red
    }
}

Write-Host "🏁 Test Battery Complete!" -ForegroundColor Green
exit $testExitCode
