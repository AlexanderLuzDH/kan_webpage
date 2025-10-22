#!/bin/bash
# KAN Shield Homepage Deployment Script
# Run this from the kan_webpage repository root

set -e

echo "📝 KAN Shield Homepage Deployment"
echo "=================================="

# Check if we're in a git repository
if [ ! -d .git ]; then
    echo "❌ Error: Not in a git repository. Run this from kan_webpage root."
    exit 1
fi

# Configure git (if not already done)
if [ -z "$(git config user.name)" ]; then
    echo "Configuring git..."
    git config user.name "KAN Shield Deploy"
    git config user.email "deploy@busleyden.com"
fi

# Stage the homepage changes
echo "📦 Staging changes..."
git add website_copy/index.html
git add website_copy/kan-shield.html
git add website_copy/kan-shield-backend.py
git add website_copy/KAN_SHIELD_DEPLOYMENT.md
git add website_copy/LAUNCH_PLAN.md

# Check if there are changes to commit
if [ -z "$(git diff --cached --name-only)" ]; then
    echo "⚠️  No changes to commit."
    exit 0
fi

# Show what will be committed
echo ""
echo "Files to be committed:"
git diff --cached --name-only
echo ""

# Commit
echo "💾 Committing changes..."
git commit -m "Add KAN Shield landing page and homepage updates

- Add kan-shield.html landing page with full feature set
- Add kan-shield-backend.py for production deployment
- Update homepage (index.html) to include KAN Shield product card
- Add deployment guide and launch plan

KAN Shield features:
- Adversarial prompt detection
- Multi-modal OCR/ASR
- Tool rails for agents
- HIPAA/GDPR/PCI compliance packs
- SIEM export (Splunk/Datadog/Elasticsearch)
- 14-day free trial, no credit card required"

# Push to main branch
echo "🚀 Pushing to GitHub..."
git push origin main

echo ""
echo "✅ Deployment successful!"
echo "🌐 Vercel will auto-deploy in ~1-2 minutes"
echo "🔗 Check: https://kan-webpage.vercel.app"
