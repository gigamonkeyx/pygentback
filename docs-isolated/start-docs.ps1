# Start VitePress Documentation Server
# This script temporarily moves conflicting config files to avoid Tailwind CSS conflicts

Write-Host "🚀 Starting PyGent Factory Documentation Server..." -ForegroundColor Green

# Check if parent vite.config.ts exists and move it temporarily
$parentViteConfig = "D:/mcp/pygent-factory/vite.config.ts"
$backupViteConfig = "D:/mcp/pygent-factory/vite.config.ts.backup"

if (Test-Path $parentViteConfig) {
    Write-Host "📦 Temporarily moving parent vite.config.ts..." -ForegroundColor Yellow
    Move-Item $parentViteConfig $backupViteConfig
}

# Start VitePress
Write-Host "📚 Starting VitePress on port 3001..." -ForegroundColor Blue
try {
    npx vitepress dev --port 3001
} finally {
    # Restore the parent vite.config.ts when VitePress stops
    if (Test-Path $backupViteConfig) {
        Write-Host "🔄 Restoring parent vite.config.ts..." -ForegroundColor Yellow
        Move-Item $backupViteConfig $parentViteConfig
    }
}

Write-Host "✅ Documentation server stopped and configs restored." -ForegroundColor Green