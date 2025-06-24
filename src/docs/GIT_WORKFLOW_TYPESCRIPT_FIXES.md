# Git Workflow for TypeScript Build Fixes

## 🎯 Purpose

This document provides the exact Git workflow used to successfully fix TypeScript build errors in the `gigamonkeyx/pygent` repository. It focuses on selective file staging and GitHub authentication patterns.

## ⚠️ Critical Prerequisites

1. **Filesystem MCP Required**: Standard file tools hang - use only Filesystem MCP
2. **GitHub PAT**: Personal Access Token with `repo` permissions
3. **Selective Staging**: Never commit all 600+ files - stage only specific files

---

## 🔄 Complete Git Workflow

### **Step 1: Repository Setup and Remote Configuration**

```bash
# Check current Git status
git status

# Add remote if not already configured
git remote add origin https://github.com/gigamonkeyx/pygent.git

# Fetch remote branches
git fetch origin

# Check remote configuration
git remote -v
```

### **Step 2: Reset Staging Area (Critical!)**

**Problem**: Repository may have 600+ files ready to commit  
**Solution**: Reset everything and start clean

```bash
# Reset all staged files
git reset

# Verify clean state
git status

# Should show: "nothing added to commit but untracked files present"
```

### **Step 3: Stage Only Required TypeScript Files**

**Files to Stage**:
- `src/services/websocket.ts` (duplicate export fix)
- `src/stores/appStore.ts` (unused parameter fix)  
- `src/vite-env.d.ts` (ImportMeta interface fix)

```bash
# Stage ONLY the 3 specific TypeScript files
git add src/services/websocket.ts src/stores/appStore.ts src/vite-env.d.ts

# Verify only these 3 files are staged
git diff --name-only --cached
```

**Expected Output**:
```
src/services/websocket.ts
src/stores/appStore.ts
src/vite-env.d.ts
```

### **Step 4: Create Descriptive Commit**

```bash
git commit -m "Fix TypeScript build errors for Cloudflare Pages deployment

- Remove duplicate EventHandler export from websocket.ts
- Fix unused parameter in appStore.ts (set, get) -> (set)
- Ensure vite-env.d.ts has proper ImportMeta interface

Resolves TypeScript compilation errors for React UI deployment."
```

**Commit Message Structure**:
- **Title**: Clear, action-oriented summary
- **Body**: Bullet points for each fix
- **Context**: Reference to deployment target (Cloudflare Pages)
- **Resolution**: What the commit accomplishes

### **Step 5: Push with GitHub PAT Authentication**

#### **5.1 Standard Push Attempt**

```bash
git push https://gigamonkeyx:[PAT]@github.com/gigamonkeyx/pygent.git master:main
```

**Replace `[PAT]` with actual Personal Access Token**

#### **5.2 Handle Push Conflicts (If Needed)**

**If you get "non-fast-forward" error**:

```bash
# Force push to override remote history
git push https://gigamonkeyx:[PAT]@github.com/gigamonkeyx/pygent.git master:main --force
```

**Example with actual PAT format**:
```bash
git push https://gigamonkeyx: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@github.com/gigamonkeyx/pygent.git master:main --force
```

---

## 🔐 GitHub Authentication Details

### **Personal Access Token (PAT) Setup**

1. **Navigate to**: https://github.com/settings/tokens
2. **Click**: "Generate new token (classic)"
3. **Permissions**: Select `repo` (Full control of private repositories)
4. **Expiration**: Set appropriate expiration date
5. **Generate**: Copy token immediately (won't be shown again)

### **PAT Usage in Git Commands**

**Format**:
```
https://username:PAT@github.com/owner/repository.git
```

**Example**:
```
https://gigamonkeyx:ghp_xxxxxxxxxxxxx@github.com/gigamonkeyx/pygent.git
```

### **Security Best Practices**

- ✅ **Use PAT instead of password** for HTTPS authentication
- ✅ **Set appropriate expiration** for tokens
- ✅ **Limit scope** to minimum required permissions
- ❌ **Never commit PAT** to repository
- ❌ **Don't share PAT** in documentation or logs

---

## 📊 Verification Commands

### **Before Staging**
```bash
# Check repository status
git status

# See what files would be committed
git diff --name-only
```

### **After Staging**
```bash
# Verify only specific files are staged
git diff --name-only --cached

# See detailed changes
git diff --cached
```

### **After Commit**
```bash
# View commit details
git log --oneline -1

# See commit content
git show HEAD
```

### **After Push**
```bash
# Verify remote tracking
git branch -vv

# Check remote status
git status
```

---

## 🚨 Common Issues and Solutions

### **Issue: "Updates were rejected because a pushed branch tip is behind"**

**Cause**: Local branch history differs from remote  
**Solution**: Use force push (with caution)

```bash
git push [remote-url] [branch] --force
```

### **Issue: "Authentication failed"**

**Cause**: Invalid or expired PAT  
**Solutions**:
1. Verify PAT is correct and not expired
2. Check PAT has `repo` permissions
3. Ensure username is correct in URL

### **Issue: "Too many files staged"**

**Cause**: Accidentally staged all repository files  
**Solution**: Reset and stage selectively

```bash
git reset
git add [specific-files-only]
```

### **Issue: "File not found" during staging**

**Cause**: Incorrect file paths  
**Solution**: Verify exact paths

```bash
# List files to verify paths
ls -la src/services/
ls -la src/stores/
ls -la src/
```

---

## 📈 Success Metrics

### **Staging Success**
- ✅ Only 3 files staged (verified with `git diff --name-only --cached`)
- ✅ No unintended files included
- ✅ All required TypeScript files included

### **Commit Success**
- ✅ Descriptive commit message
- ✅ Commit hash generated
- ✅ Changes properly recorded

### **Push Success**
- ✅ No authentication errors
- ✅ Remote repository updated
- ✅ Commit visible on GitHub

### **Overall Success**
- ✅ GitHub files match local fixes
- ✅ Cloudflare Pages build succeeds
- ✅ TypeScript errors resolved

---

## 🔄 Workflow Summary

```mermaid
graph TD
    A[Check Git Status] --> B[Reset Staging Area]
    B --> C[Stage Only 3 TypeScript Files]
    C --> D[Verify Staged Files]
    D --> E[Create Descriptive Commit]
    E --> F[Push with PAT Authentication]
    F --> G{Push Successful?}
    G -->|No| H[Force Push]
    G -->|Yes| I[Verify on GitHub]
    H --> I
    I --> J[Monitor Cloudflare Pages]
    J --> K[Success!]
```

---

## 📝 Quick Command Reference

### **Essential Commands**
```bash
# Reset and start clean
git reset

# Stage specific files
git add src/services/websocket.ts src/stores/appStore.ts src/vite-env.d.ts

# Verify staging
git diff --name-only --cached

# Commit with message
git commit -m "[descriptive message]"

# Push with PAT
git push https://username:PAT@github.com/owner/repo.git branch:target

# Force push if needed
git push https://username:PAT@github.com/owner/repo.git branch:target --force
```

### **Verification Commands**
```bash
# Check what's staged
git diff --name-only --cached

# View commit details
git log --oneline -1

# Check remote status
git status
```

---

## 🎯 For AI Assistants

**Critical Workflow Points**:

1. **Always reset first** - prevents committing 600+ files
2. **Stage selectively** - only the 3 TypeScript files
3. **Use PAT authentication** - required for GitHub HTTPS
4. **Force push when necessary** - handles history conflicts
5. **Verify on GitHub** - confirm changes are live

**Success Pattern**:
`Reset → Selective Stage → Verify → Commit → PAT Push → Force if Needed → Verify`

**Key Files**:
- `src/services/websocket.ts`
- `src/stores/appStore.ts`  
- `src/vite-env.d.ts`

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-04  
**Tested On**: gigamonkeyx/pygent repository  
**Commit Reference**: 0d10511  
**Status**: ✅ Verified Working