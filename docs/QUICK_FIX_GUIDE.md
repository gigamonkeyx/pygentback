# Quick Deployment Fix Guide

## 🚀 MANUAL STEPS TO FIX DEPLOYMENT

### Step 1: Start Backend Server
1. Open **Command Prompt** or **PowerShell** as Administrator
2. Navigate to: `d:\mcp\pygent-factory`
3. Run: `start-backend-quick.bat`
4. **Expected output**: "Server started on http://0.0.0.0:8000"
5. **Leave this terminal open** - backend must keep running

### Step 2: Start Cloudflare Tunnel
1. Open **another Command Prompt** or **PowerShell** 
2. Navigate to: `d:\mcp\pygent-factory`
3. Run: `start-tunnel-quick.bat`
4. **Expected output**: Tunnel connection established
5. **Leave this terminal open** - tunnel must keep running

### Step 3: Verify Everything Works
1. **Backend Test**: Open browser to http://localhost:8000/health
2. **Tunnel Test**: Open browser to https://ws.timpayne.net/health  
3. **Deployed UI**: Visit https://pygent.pages.dev/
4. **WebSocket**: Should connect automatically (check browser console)

## 🔧 TROUBLESHOOTING

### If Backend Won't Start:
```bash
cd d:\mcp\pygent-factory
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --mode server
```

### If Tunnel Won't Start:
- Check internet connection
- Verify Cloudflare credentials
- Try: `.\cloudflared-new.exe tunnel list`

### If WebSocket Still Fails:
- Wait 30 seconds for tunnel to fully connect
- Refresh the deployed UI page
- Check both terminals are still running

## ✅ SUCCESS INDICATORS

You'll know it's working when:
- Backend terminal shows "Server started"
- Tunnel terminal shows "Connection established"
- https://pygent.pages.dev/ loads without WebSocket errors
- Browser console shows "WebSocket connected successfully"

## 📋 KEEP RUNNING

**Important**: Both terminals must stay open and running:
- **Terminal 1**: Backend server (port 8000)
- **Terminal 2**: Cloudflare tunnel (routing)

Close either one and the deployment will break again.

Ready to test! 🎯
