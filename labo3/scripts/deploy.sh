#!/bin/bash
# deploy.sh - Simple deployment script for ML experiments
# Can be run from anywhere within the git repository

set -e

# Find git repository root first
echo "🔍 Detecting git repository..."
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -z "$GIT_ROOT" ]; then
    echo "❌ Not in a git repository! Please run from within a git repository."
    exit 1
fi

echo "📁 Git repository root: $GIT_ROOT"
ORIGINAL_DIR=$(pwd)

# Load configuration
CONFIG_FILE="project_config.yml"

# First check current directory, then git root
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$GIT_ROOT/$CONFIG_FILE" ]; then
        CONFIG_FILE="$GIT_ROOT/$CONFIG_FILE"
        echo "📋 Using config from git root: $CONFIG_FILE"
    else
        echo "❌ project_config.yml not found in current directory or git root!"
        echo "   Current: $(pwd)/$CONFIG_FILE"
        echo "   Git root: $GIT_ROOT/$CONFIG_FILE"
        exit 1
    fi
fi

# Parse YAML config (requires yq)
if ! command -v yq &> /dev/null; then
    echo "Installing yq..."
    if command -v brew &> /dev/null; then
        brew install yq
    else
        # Fallback for Linux systems without brew
        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
        sudo chmod +x /usr/local/bin/yq
    fi
fi

PROJECT_ID=$(yq '.gcp.project_id' $CONFIG_FILE)
BUCKET_NAME=$(yq '.gcp.bucket_name' $CONFIG_FILE)
ORCHESTRATOR_VM=$(yq '.orchestrator.vm_name' $CONFIG_FILE)
ZONE=$(yq '.gcp.zone' $CONFIG_FILE)

echo "🚀 Starting deployment for project: $(yq '.project.name' $CONFIG_FILE)"

# 1. Git operations - Commit and push latest changes
echo "📝 Committing and pushing latest changes..."

# Change to git root for git operations
cd "$GIT_ROOT"

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo "📝 Found uncommitted changes, committing..."
    git add .
    git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S') - Experiment deployment" || echo "No changes to commit"
else
    echo "✅ No uncommitted changes found"
fi

# Push to remote
echo "🔄 Pushing to remote repository..."
git push origin main || git push origin master || {
    echo "❌ Failed to push to remote repository"
    echo "Please check your git remote configuration and try again"
    exit 1
}

echo "✅ Code pushed to remote repository"

# Return to original directory for the rest of the script
cd "$ORIGINAL_DIR"

# 2. Authenticate with service account
echo "🔑 Authenticating with GCP..."

# Check for service account in current directory or git root
SERVICE_ACCOUNT_FILE="service-account.json"
if [ ! -f "$SERVICE_ACCOUNT_FILE" ]; then
    if [ -f "$GIT_ROOT/$SERVICE_ACCOUNT_FILE" ]; then
        SERVICE_ACCOUNT_FILE="$GIT_ROOT/$SERVICE_ACCOUNT_FILE"
        echo "🔑 Using service account from git root: $SERVICE_ACCOUNT_FILE"
    else
        echo "❌ service-account.json not found in current directory or git root!"
        echo "   Current: $(pwd)/$SERVICE_ACCOUNT_FILE"
        echo "   Git root: $GIT_ROOT/$SERVICE_ACCOUNT_FILE"
        exit 1
    fi
fi

gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE"
gcloud config set project $PROJECT_ID

# 3. Build package (skip this - we'll install directly on workers)
echo "📦 Skipping package build - workers will install from source"

# 4. Upload artifacts to GCS
echo "☁️ Uploading configuration to GCS..."

# Only upload config file - no more wheel files
gsutil cp "$CONFIG_FILE" gs://$BUCKET_NAME/config/

echo "💡 Note: Workers will install package from source using uv"

# Upload .env file if it exists (check both locations)
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "$GIT_ROOT/.env" ]; then
    ENV_FILE="$GIT_ROOT/.env"
fi

if [ -n "$ENV_FILE" ]; then
    echo "📄 Uploading .env file: $ENV_FILE"
    gsutil cp "$ENV_FILE" gs://$BUCKET_NAME/config/
fi

# 5. Upload data files
echo "📁 Uploading data files..."
yq '.paths.data_files[]' "$CONFIG_FILE" | while read -r file; do
    # Check file in current directory first, then git root
    DATA_FILE=""
    if [ -f "$file" ]; then
        DATA_FILE="$file"
    elif [ -f "$GIT_ROOT/$file" ]; then
        DATA_FILE="$GIT_ROOT/$file"
    fi
    
    if [ -n "$DATA_FILE" ]; then
        echo "📤 Uploading: $DATA_FILE"
        gsutil cp "$DATA_FILE" gs://$BUCKET_NAME/$file
    else
        echo "⚠️ Data file not found: $file"
        echo "   Checked: $(pwd)/$file"
        echo "   Checked: $GIT_ROOT/$file"
    fi
done

# 6. Check if VM is running
echo "🔍 Checking orchestrator VM status..."
VM_STATUS=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" != "RUNNING" ]; then
    if [ "$VM_STATUS" = "NOT_FOUND" ]; then
        echo "❌ Orchestrator VM not found! Please create it first."
        exit 1
    else
        echo "🚀 Starting orchestrator VM..."
        gcloud compute instances start $ORCHESTRATOR_VM --zone=$ZONE
        
        echo "⏳ Waiting for VM to be ready..."
        sleep 30
    fi
fi

# 7. Trigger deployment
echo "🎯 Triggering deployment..."
DEPLOY_ID=$(date '+%Y%m%d_%H%M%S')

# Create deployment signal
gcloud compute instances add-metadata $ORCHESTRATOR_VM \
    --zone=$ZONE \
    --metadata deploy-trigger="$DEPLOY_ID"

echo "✅ Deployment triggered with ID: $DEPLOY_ID"
echo ""
echo "📊 Monitor deployment with:"
echo "  gcloud compute ssh $ORCHESTRATOR_VM --zone=$ZONE --command='tmux attach-session -t orchestrator'"
echo ""
echo "📈 Check MLflow at:"
echo "  http://$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):5000"
echo ""
echo "📋 View logs with:"
echo "  gcloud compute instances get-serial-port-output $ORCHESTRATOR_VM --zone=$ZONE"