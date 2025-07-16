# Fix Docker credentials issue
Write-Host "🔧 Fixing Docker credentials..." -ForegroundColor Yellow

$dockerConfigPath = "$env:USERPROFILE\.docker\config.json"

# Backup existing config if it exists
if (Test-Path $dockerConfigPath) {
    Copy-Item $dockerConfigPath "$dockerConfigPath.backup"
    Write-Host "✅ Backed up existing config" -ForegroundColor Green
}

# Create new config without credential helpers for public registries
$config = @{
    "auths" = @{}
    "credsStore" = ""
    "credHelpers" = @{}
}

# Convert to JSON and save
$config | ConvertTo-Json -Depth 10 | Set-Content $dockerConfigPath

Write-Host "✅ Updated Docker config to disable credential helpers" -ForegroundColor Green

# Test Docker pull
Write-Host "🧪 Testing Docker pull..." -ForegroundColor Yellow
try {
    & "D:\Docker\resources\bin\docker.exe" pull hello-world
    Write-Host "✅ Docker pull test successful!" -ForegroundColor Green
} catch {
    Write-Warning "⚠️ Docker pull test failed: $_"
}

Write-Host "🎉 Docker credentials fixed!" -ForegroundColor Green
