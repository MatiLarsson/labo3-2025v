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

# Check multiple locations for config file
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$GIT_ROOT/$CONFIG_FILE" ]; then
        CONFIG_FILE="$GIT_ROOT/$CONFIG_FILE"
        echo "📋 Using config from git root: $CONFIG_FILE"
    elif [ -f "$GIT_ROOT/labo3/$CONFIG_FILE" ]; then
        CONFIG_FILE="$GIT_ROOT/labo3/$CONFIG_FILE"
        echo "📋 Using config from labo3 subdirectory: $CONFIG_FILE"
    elif [ -f "labo3/$CONFIG_FILE" ]; then
        CONFIG_FILE="labo3/$CONFIG_FILE"
        echo "📋 Using config from labo3 subdirectory: $CONFIG_FILE"
    else
        echo "❌ project_config.yml not found in any expected location!"
        echo "   Current: $(pwd)/$CONFIG_FILE"
        echo "   Git root: $GIT_ROOT/$CONFIG_FILE"
        echo "   Labo3: $GIT_ROOT/labo3/$CONFIG_FILE"
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

gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE" --quiet
gcloud config set project $PROJECT_ID --quiet

# 3. Build package (skip this - we'll install directly on workers)
echo "📦 Skipping package build - workers will install from source"

# 4. Upload artifacts to GCS
echo "☁️ Uploading configuration to GCS..."

# Only upload config file - no more wheel files
gsutil cp "$CONFIG_FILE" gs://$BUCKET_NAME/config/

echo "💡 Note: Workers will install package from source using uv"

# Upload .env file with updated MLflow URI (check multiple locations)
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "$GIT_ROOT/.env" ]; then
    ENV_FILE="$GIT_ROOT/.env"
elif [ -f "$GIT_ROOT/labo3/.env" ]; then
    ENV_FILE="$GIT_ROOT/labo3/.env"
elif [ -f "labo3/.env" ]; then
    ENV_FILE="labo3/.env"
fi

if [ -n "$ENV_FILE" ]; then
    echo "📄 Updating .env file with current orchestrator IP..."
    
    # Get current orchestrator VM IP
    ORCHESTRATOR_IP=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")
    
    if [ -n "$ORCHESTRATOR_IP" ]; then
        echo "📡 Current orchestrator IP: $ORCHESTRATOR_IP"
        
        # Create temporary .env file with updated MLflow URI
        cp "$ENV_FILE" .env.tmp
        
        # Remove old MLFLOW_TRACKING_URI line and add new one
        grep -v "MLFLOW_TRACKING_URI=" .env.tmp > .env.updated || cp .env.tmp .env.updated
        echo "MLFLOW_TRACKING_URI=http://$ORCHESTRATOR_IP:5000" >> .env.updated
        
        echo "📄 Uploading updated .env file: $ENV_FILE"
        gsutil cp .env.updated gs://$BUCKET_NAME/config/.env
        
        # Clean up temporary files
        rm -f .env.tmp .env.updated
        
        echo "✅ MLflow URI updated to: http://$ORCHESTRATOR_IP:5000"
    else
        echo "⚠️ Could not get orchestrator IP, uploading .env as-is"
        gsutil cp "$ENV_FILE" gs://$BUCKET_NAME/config/
    fi
else
    echo "⚠️ No .env file found to upload"
fi

# 5. Upload data files (with existence check)
echo "📁 Uploading data files..."
yq '.paths.data_files[]' "$CONFIG_FILE" | while read -r file; do
    # Check file in multiple locations: current dir, git root, and labo3 subdirectory
    DATA_FILE=""
    if [ -f "$file" ]; then
        DATA_FILE="$file"
    elif [ -f "$GIT_ROOT/$file" ]; then
        DATA_FILE="$GIT_ROOT/$file"
    elif [ -f "$GIT_ROOT/labo3/$file" ]; then
        DATA_FILE="$GIT_ROOT/labo3/$file"
    elif [ -f "labo3/$file" ]; then
        DATA_FILE="labo3/$file"
    fi
    
    if [ -n "$DATA_FILE" ]; then
        GCS_PATH="gs://$BUCKET_NAME/$file"
        
        # Check if file already exists in GCS
        if gsutil -q stat "$GCS_PATH" 2>/dev/null; then
            # File exists, compare sizes (much faster than hash)
            LOCAL_SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
            REMOTE_SIZE=$(gsutil du "$GCS_PATH" | awk '{print $1}')
            
            if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
                echo "⏭️  Skipping (same size): $file ($LOCAL_SIZE bytes)"
                continue
            else
                echo "🔄 Updating (size changed): $file (local: $LOCAL_SIZE, remote: $REMOTE_SIZE bytes)"
            fi
        else
            echo "📤 Uploading (new): $file"
        fi
        
        gsutil cp "$DATA_FILE" "$GCS_PATH"
    else
        echo "⚠️ Data file not found: $file"
        echo "   Checked: $(pwd)/$file"
        echo "   Checked: $GIT_ROOT/$file"
        echo "   Checked: $GIT_ROOT/labo3/$file"
        echo "   Checked: labo3/$file"
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
echo "  gcloud compute ssh $ORCHESTRATOR_VM --zone=$ZONE --command='sudo tmux attach-session -t orchestrator'"
echo ""
echo "📈 Check MLflow at:"
echo "  http://$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):5000"
echo ""
echo "📋 View logs with:"
echo "  gcloud compute instances get-serial-port-output $ORCHESTRATOR_VM --zone=$ZONE"