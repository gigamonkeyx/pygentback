#!/usr/bin/env node

/**
 * Sync-to-Frontend Script for PyGent Factory Hybrid Documentation
 * 
 * This script copies built VitePress documentation from the docs/dist directory
 * to the frontend serving location for seamless integration with the React UI.
 */

import { fileURLToPath } from 'url';
import { dirname, join } from 'path';
import { existsSync, mkdirSync, cpSync, rmSync } from 'fs';
import { execSync } from 'child_process';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Define paths
const DOCS_ROOT = join(__dirname, '..');
const DOCS_DIST = join(DOCS_ROOT, 'dist');
const PROJECT_ROOT = join(DOCS_ROOT, '..', '..');
const FRONTEND_PUBLIC = join(PROJECT_ROOT, 'pygent-repo', 'public');
const FRONTEND_DOCS_TARGET = join(FRONTEND_PUBLIC, 'docs');

console.log('🚀 PyGent Factory Documentation Sync');
console.log('=====================================');

function validatePaths() {
  console.log('📁 Validating paths...');
  
  if (!existsSync(DOCS_DIST)) {
    console.error('❌ Documentation build not found. Please run "npm run build" first.');
    console.log(`   Expected: ${DOCS_DIST}`);
    process.exit(1);
  }
  
  if (!existsSync(FRONTEND_PUBLIC)) {
    console.error('❌ Frontend public directory not found.');
    console.log(`   Expected: ${FRONTEND_PUBLIC}`);
    console.log('   Make sure you are running this from the correct project structure.');
    process.exit(1);
  }
  
  console.log('✅ All paths validated');
}

function cleanTarget() {
  console.log('🧹 Cleaning target directory...');
  
  if (existsSync(FRONTEND_DOCS_TARGET)) {
    rmSync(FRONTEND_DOCS_TARGET, { recursive: true, force: true });
    console.log('✅ Cleaned existing documentation');
  }
}

function copyDocumentation() {
  console.log('📋 Copying documentation files...');
  
  try {
    // Create target directory
    mkdirSync(FRONTEND_DOCS_TARGET, { recursive: true });
    
    // Copy all built documentation files
    cpSync(DOCS_DIST, FRONTEND_DOCS_TARGET, { 
      recursive: true,
      force: true,
      preserveTimestamps: true
    });
    
    console.log('✅ Documentation copied successfully');
    
    // Log some statistics
    const stats = execSync(`find "${FRONTEND_DOCS_TARGET}" -type f | wc -l`, { encoding: 'utf8' }).trim();
    console.log(`📊 Copied ${stats} files`);
    
  } catch (error) {
    console.error('❌ Failed to copy documentation:', error.message);
    process.exit(1);
  }
}

function createRouteManifest() {
  console.log('📝 Creating route manifest...');
  
  try {
    // Create a simple manifest for the frontend to understand available routes
    const manifest = {
      timestamp: new Date().toISOString(),
      baseUrl: '/docs/',
      routes: [
        '/',
        '/getting-started/introduction',
        '/getting-started/quick-start',
        '/concepts/architecture',
        '/concepts/mcp-protocol',
        '/api/',
        '/guides/',
        '/advanced/'
      ]
    };
    
    const manifestPath = join(FRONTEND_DOCS_TARGET, 'manifest.json');
    require('fs').writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    
    console.log('✅ Route manifest created');
    
  } catch (error) {
    console.error('⚠️  Warning: Could not create route manifest:', error.message);
  }
}

function validateSync() {
  console.log('🔍 Validating sync...');
  
  const indexPath = join(FRONTEND_DOCS_TARGET, 'index.html');
  if (!existsSync(indexPath)) {
    console.error('❌ Sync validation failed: index.html not found');
    process.exit(1);
  }
  
  const assetsPath = join(FRONTEND_DOCS_TARGET, 'assets');
  if (!existsSync(assetsPath)) {
    console.error('❌ Sync validation failed: assets directory not found');
    process.exit(1);
  }
  
  console.log('✅ Sync validation passed');
}

function main() {
  try {
    validatePaths();
    cleanTarget();
    copyDocumentation();
    createRouteManifest();
    validateSync();
    
    console.log('');
    console.log('🎉 Documentation sync completed successfully!');
    console.log(`📍 Documentation available at: ${FRONTEND_DOCS_TARGET}`);
    console.log('🔗 Frontend can now serve docs at /docs/ route');
    
  } catch (error) {
    console.error('💥 Sync failed:', error.message);
    process.exit(1);
  }
}

// Run the sync
main();
