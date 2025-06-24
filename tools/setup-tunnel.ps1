# PyGent Factory Cloudflared Tunnel Setup Script
# This script sets up secure tunnels for the PyGent Factory backend

Write-Host "🚇 Setting up Cloudflared Tunnel for PyGent Factory..." -ForegroundColor Green

# Check if cloudflared is installed
try {
    $cloudflaredVersion = cloudflared --version
    Write-Host "✅ Cloudflared found: $cloudflaredVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Cloudflared not found. Please install it first." -ForegroundColor Red
    Write-Host "Download from: https://github.com/cloudflare/cloudflared/releases" -ForegroundColor Yellow
    exit 1
}

# Create logs directory
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
    Write-Host "📁 Created logs directory" -ForegroundColor Yellow
}

# Authenticate with Cloudflare (if not already done)
Write-Host "🔐 Checking Cloudflare authentication..." -ForegroundColor Yellow
try {
    $authCheck = cloudflared tunnel list 2>&1
    if ($authCheck -like "*not authenticated*" -or $authCheck -like "*login*") {
        Write-Host "⚠️ Not authenticated with Cloudflare. Please run:" -ForegroundColor Yellow
        Write-Host "   cloudflared tunnel login" -ForegroundColor White
        Write-Host "Then run this script again." -ForegroundColor Yellow
        exit 1
    }
    Write-Host "✅ Cloudflare authentication verified" -ForegroundColor Green
} catch {
    Write-Host "⚠️ Could not verify authentication. Please ensure you're logged in:" -ForegroundColor Yellow
    Write-Host "   cloudflared tunnel login" -ForegroundColor White
}

# Create tunnel if it doesn't exist
$tunnelName = "pygent-factory-api"
Write-Host "🚇 Checking for existing tunnel: $tunnelName" -ForegroundColor Yellow

try {
    $existingTunnel = cloudflared tunnel list | Select-String $tunnelName
    if ($existingTunnel) {
        Write-Host "✅ Tunnel '$tunnelName' already exists" -ForegroundColor Green
    } else {
        Write-Host "🆕 Creating new tunnel: $tunnelName" -ForegroundColor Yellow
        cloudflared tunnel create $tunnelName
        Write-Host "✅ Tunnel created successfully" -ForegroundColor Green
    }
} catch {
    Write-Host "❌ Failed to create tunnel: $_" -ForegroundColor Red
    exit 1
}

# Get tunnel ID
try {
    $tunnelInfo = cloudflared tunnel list | Select-String $tunnelName
    if ($tunnelInfo) {
        $tunnelId = ($tunnelInfo -split '\s+')[0]
        Write-Host "🆔 Tunnel ID: $tunnelId" -ForegroundColor Cyan
        
        # Update config file with correct credentials path
        $credentialsPath = "$env:USERPROFILE\.cloudflared\$tunnelId.json"
        if (Test-Path $credentialsPath) {
            Write-Host "📋 Updating configuration with credentials path..." -ForegroundColor Yellow
            $configContent = Get-Content "cloudflared-config.yml" -Raw
            $configContent = $configContent -replace "/path/to/credentials.json", $credentialsPath
            $configContent = $configContent -replace "tunnel: pygent-factory-api", "tunnel: $tunnelId"
            Set-Content "cloudflared-config.yml" $configContent
            Write-Host "✅ Configuration updated" -ForegroundColor Green
        } else {
            Write-Host "⚠️ Credentials file not found at: $credentialsPath" -ForegroundColor Yellow
        }
    }
} catch {
    Write-Host "❌ Failed to get tunnel information: $_" -ForegroundColor Red
}

Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Configure DNS records in Cloudflare dashboard:" -ForegroundColor White
Write-Host "     - api.timpayne.net CNAME $tunnelId.cfargotunnel.com" -ForegroundColor White
Write-Host "     - ws.timpayne.net CNAME $tunnelId.cfargotunnel.com" -ForegroundColor White
Write-Host "  2. Start the tunnel with: .\start-tunnel.ps1" -ForegroundColor White
Write-Host "  3. Test the API at: https://api.timpayne.net/api/v1/health" -ForegroundColor White

Write-Host "🎉 Tunnel setup complete!" -ForegroundColor Green
