# Simple PyGent Factory UI Deployment Preparation

Write-Host "🚀 PyGent Factory UI Deployment Preparation" -ForegroundColor Green

# Navigate to UI directory
Set-Location "src\ui"

# Build the UI
Write-Host "📦 Building UI..." -ForegroundColor Yellow
npm run build

# Go back to root
Set-Location "..\.."

# Create deployment directory
$deployDir = "pygent-ui-deploy"
if (Test-Path $deployDir) {
    Remove-Item -Recurse -Force $deployDir
}
New-Item -ItemType Directory -Name $deployDir

# Copy essential files
Write-Host "📋 Copying files..." -ForegroundColor Yellow
Copy-Item -Recurse "src\ui\src" "$deployDir\src"
Copy-Item "src\ui\index.html" "$deployDir\"
Copy-Item "src\ui\package.json" "$deployDir\"
Copy-Item "src\ui\package-lock.json" "$deployDir\"
Copy-Item "src\ui\vite.config.ts" "$deployDir\"
Copy-Item "src\ui\tsconfig.json" "$deployDir\"
Copy-Item "src\ui\tailwind.config.js" "$deployDir\"
Copy-Item "src\ui\postcss.config.js" "$deployDir\"

# Create production README
$readme = @"
# PyGent Factory UI

React application for PyGent Factory deployed on Cloudflare Pages.

## Development
\`\`\`bash
npm install
npm run dev
\`\`\`

## Build
\`\`\`bash
npm run build
\`\`\`
"@

Set-Content -Path "$deployDir\README.md" -Value $readme

# Create _redirects for SPA
Set-Content -Path "$deployDir\_redirects" -Value "/*    /index.html   200"

Write-Host "✅ Deployment directory created: $deployDir" -ForegroundColor Green
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "  1. cd $deployDir" -ForegroundColor White
Write-Host "  2. git init" -ForegroundColor White
Write-Host "  3. git add ." -ForegroundColor White
Write-Host "  4. git commit -m 'Initial commit'" -ForegroundColor White
Write-Host "  5. git remote add origin https://github.com/gigamonkeyx/pygent.git" -ForegroundColor White
Write-Host "  6. git push -u origin main" -ForegroundColor White
