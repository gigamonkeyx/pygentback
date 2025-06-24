# Test workflow creation with PowerShell
$body = @{
    query = "quantum computing applications"
    analysis_model = "deepseek-r1:8b"
    max_papers = 5
    analysis_depth = 1
} | ConvertTo-Json

Write-Host "Testing workflow creation..."
Write-Host "Body: $body"

try {
    $response = Invoke-RestMethod -Uri "http://localhost:8080/api/v1/workflows/research-analysis" -Method POST -Body $body -ContentType "application/json"
    Write-Host "SUCCESS: Workflow created!"
    Write-Host "Workflow ID: $($response.workflow_id)"
    Write-Host "Status: $($response.status)"
    Write-Host "Message: $($response.message)"
} catch {
    Write-Host "ERROR: $($_.Exception.Message)"
    Write-Host "Response: $($_.Exception.Response)"
}
