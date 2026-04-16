# Test website connectivity
Write-Host "=== Testing Sleep Portal Website ===" -ForegroundColor Cyan

# Test DNS resolution
Write-Host ""
Write-Host "1. DNS Resolution:" -ForegroundColor Yellow
$dns = [System.Net.Dns]::GetHostAddresses("sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com")
Write-Host "   Resolved IPs: $($dns.IPAddressToString -join ', ')"

# Test HTTP connection
Write-Host ""
Write-Host "2. HTTP Connection Test:" -ForegroundColor Yellow
try {
    $response = Invoke-WebRequest -Uri "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/" -TimeoutSec 10 -UseBasicParsing
    Write-Host "   Status: SUCCESS" -ForegroundColor Green
    Write-Host "   HTTP Code: $($response.StatusCode)"
    Write-Host "   Content Length: $($response.Content.Length) bytes"
    Write-Host "   Preview: $($response.Content.Substring(0, [Math]::Min(150, $response.Content.Length)))"
} catch {
    Write-Host "   Status: FAILED" -ForegroundColor Red
    Write-Host "   Error: $($_.Exception.Message)"
}

# Test Health endpoint
Write-Host ""
Write-Host "3. Health Endpoint Test:" -ForegroundColor Yellow
try {
    $health = Invoke-WebRequest -Uri "http://sleep-portal-alb-1369421469.ap-southeast-1.elb.amazonaws.com/api/v1/health/" -TimeoutSec 5 -UseBasicParsing
    Write-Host "   Health Status: $($health.Content)" -ForegroundColor Green
} catch {
    Write-Host "   Health Check Failed: $($_.Exception.Message)" -ForegroundColor Red
}

Write-Host ""
Write-Host "==================================" -ForegroundColor Cyan
