#!/bin/bash

# Setup GitHub repository for custom-object-detector
# Run this script after training your model

echo "============================================"
echo "Custom Object Detector - GitHub Setup"
echo "============================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "Initializing git repository..."
    git init
fi

# Add all files
echo "Adding files to git..."
git add .

# Create initial commit
echo "Creating initial commit..."
git commit -m "Initial commit: Custom Faster R-CNN object detector

- ResNet-18 backbone with ImageNet pretrained weights
- Mixed precision training support
- Real-time inference (~35 FPS)
- 5 classes: Person, Car, Bicycle, Speed Limit 30, Stop Sign
- Comprehensive training with validation mAP tracking"

echo ""
echo "============================================"
echo "NEXT STEPS:"
echo "============================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   https://github.com/new"
echo ""
echo "2. Name it 'custom-object-detector' (or your preferred name)"
echo ""
echo "3. Run these commands (replace YOUR_USERNAME):"
echo ""
echo "   git remote add origin https://github.com/YOUR_USERNAME/custom-object-detector.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "4. Update the GitHub link in REPORT.md"
echo ""
echo "============================================"
