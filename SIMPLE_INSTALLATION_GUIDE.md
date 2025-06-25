# PyGent Factory - Simple Installation Guide

## 🎯 **OVERVIEW**

This guide will help you install and run PyGent Factory in **under 30 minutes**. No technical expertise required!

## 📋 **WHAT YOU NEED**

### **System Requirements**
- **Operating System**: Windows 10/11, macOS, or Linux
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 10GB free space
- **Internet**: Stable internet connection for AI services

### **Software Prerequisites**
- **Docker Desktop**: For running the application
- **Web Browser**: Chrome, Firefox, Safari, or Edge

## 🚀 **INSTALLATION STEPS**

### **Step 1: Install Docker Desktop**

#### **Windows/Mac:**
1. Go to [docker.com/products/docker-desktop](https://docker.com/products/docker-desktop)
2. Download Docker Desktop for your operating system
3. Run the installer and follow the setup wizard
4. **Important**: Enable WSL 2 integration when prompted (Windows only)
5. Restart your computer when installation completes

#### **Verify Docker Installation:**
```bash
# Open Command Prompt/Terminal and run:
docker --version
# Should show: Docker version 20.x.x or higher
```

### **Step 2: Download PyGent Factory**

#### **Option A: Download ZIP (Easiest)**
1. Go to the PyGent Factory repository
2. Click "Code" → "Download ZIP"
3. Extract the ZIP file to your desired location (e.g., `C:\PyGentFactory`)

#### **Option B: Git Clone (If you have Git)**
```bash
git clone https://github.com/your-org/pygent-factory.git
cd pygent-factory
```

### **Step 3: Configure Environment**

1. **Navigate to the PyGent Factory folder**
2. **Copy the example environment file:**
   - Find the file named `.env.example`
   - Copy it and rename the copy to `.env`

3. **Edit the `.env` file** (use Notepad or any text editor):
   ```env
   # AI Service Configuration (Optional - uses free local AI by default)
   OPENAI_API_KEY=your_openai_key_here
   OPENROUTER_API_KEY=your_openrouter_key_here
   
   # Database (Leave as default for local installation)
   DATABASE_URL=postgresql://postgres:postgres@postgres:5432/pygent_factory
   
   # Application Settings (Leave as default)
   DEBUG=true
   LOG_LEVEL=INFO
   ```

### **Step 4: Start PyGent Factory**

#### **Open Command Prompt/Terminal in the PyGent Factory folder:**

**Windows:**
- Right-click in the folder → "Open in Terminal" or "Open Command Prompt here"

**Mac/Linux:**
- Open Terminal and navigate to the folder: `cd /path/to/pygent-factory`

#### **Run the startup command:**
```bash
docker-compose up -d
```

**What this does:**
- Downloads and starts all required services
- Sets up the database automatically
- Starts the web interface
- **First run takes 5-10 minutes** (downloading components)

### **Step 5: Access PyGent Factory**

1. **Wait for startup to complete** (watch the terminal output)
2. **Open your web browser**
3. **Go to**: `http://localhost:3000`
4. **You should see the PyGent Factory interface!**

## ✅ **VERIFICATION CHECKLIST**

### **Check All Services Are Running:**
```bash
docker-compose ps
```

**You should see:**
- ✅ `pygent-frontend` (Running)
- ✅ `pygent-api` (Running)  
- ✅ `postgres` (Running)
- ✅ `redis` (Running)

### **Test the Interface:**
1. **Frontend**: `http://localhost:3000` - Should show the main interface
2. **API**: `http://localhost:8080/docs` - Should show API documentation
3. **Health Check**: `http://localhost:8080/health` - Should return "OK"

## 🔧 **COMMON ISSUES & SOLUTIONS**

### **Issue: Docker Desktop won't start**
**Solution:**
1. Restart your computer
2. Ensure virtualization is enabled in BIOS
3. On Windows: Ensure WSL 2 is installed and updated

### **Issue: "Port already in use" error**
**Solution:**
```bash
# Stop any existing services
docker-compose down
# Wait 30 seconds, then try again
docker-compose up -d
```

### **Issue: Services won't start**
**Solution:**
```bash
# Check Docker is running
docker --version
# Restart Docker Desktop application
# Try the startup command again
```

### **Issue: Can't access the web interface**
**Solution:**
1. Wait 2-3 minutes for full startup
2. Check if services are running: `docker-compose ps`
3. Try refreshing your browser
4. Clear browser cache and try again

## 🎯 **FIRST STEPS AFTER INSTALLATION**

### **1. Create Your First Agent**
- Click "Create New Agent" in the interface
- Choose "Research Assistant" template
- Give it a name and description
- Click "Create"

### **2. Test Basic Functionality**
- Try the "Quick Research" feature
- Upload a document for analysis
- Ask a question and see the AI response

### **3. Explore Features**
- Browse the agent marketplace
- Check out example workflows
- Review the documentation section

## 📞 **GETTING HELP**

### **If You Need Assistance:**
1. **Check the logs**: `docker-compose logs`
2. **Restart services**: `docker-compose restart`
3. **Contact Support**: Include your system info and error messages

### **System Information to Provide:**
```bash
# Run these commands and share the output:
docker --version
docker-compose --version
docker-compose ps
```

## 🎉 **SUCCESS!**

**Congratulations! PyGent Factory is now running on your system.**

**Next Steps:**
- Explore the user interface
- Create your first AI agent
- Try the research and analysis features
- Review the user documentation for advanced features

**You're ready to start building intelligent AI solutions!**

---

**Need help? Contact our support team with your installation details and we'll get you up and running quickly.**
