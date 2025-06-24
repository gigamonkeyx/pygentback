#!/bin/bash

echo "Starting PyGent Factory Documentation..."
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed or not in PATH"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "package.json" ]; then
    echo "Error: package.json not found"
    echo "Please run this script from the docs directory"
    exit 1
fi

# Install dependencies if node_modules doesn't exist
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        exit 1
    fi
fi

# Start the development server
echo "Starting VitePress development server..."
echo "Documentation will be available at: http://localhost:3000"
echo
echo "Press Ctrl+C to stop the server"
echo

npm run dev