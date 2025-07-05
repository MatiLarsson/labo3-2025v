#!/bin/bash
# deploy.sh - Simple deployment script for ML experiments
# Can be run from anywhere within the git repository

set -e

# Suppress urllib3 warnings
export PYTHONWARNINGS="ignore:urllib3"

# Find git repository root first
echo "🔍 Detecting git repository..."
GIT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || echo "")
if [ -z "$GIT_ROOT" ]; then
    echo "❌ Not in a git repository! Please run from within a git repository."
    exit 1
fi

ORIGINAL_DIR=$(pwd)

# Load configuration
CONFIG_FILE="project_config.yml"

# Check multiple locations for config file
if [ ! -f "$CONFIG_FILE" ]; then
    if [ -f "$GIT_ROOT/$CONFIG_FILE" ]; then
        CONFIG_FILE="$GIT_ROOT/$CONFIG_FILE"
    elif [ -f "$GIT_ROOT/labo3/$CONFIG_FILE" ]; then
        CONFIG_FILE="$GIT_ROOT/labo3/$CONFIG_FILE"
    elif [ -f "labo3/$CONFIG_FILE" ]; then
        CONFIG_FILE="labo3/$CONFIG_FILE"
    else
        echo "❌ project_config.yml not found!"
        exit 1
    fi
fi

# Parse YAML config (requires yq)
if ! command -v yq &> /dev/null; then
    echo "📦 Installing yq..."
    if command -v brew &> /dev/null; then
        brew install yq > /dev/null 2>&1
    else
        sudo wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64 2>/dev/null
        sudo chmod +x /usr/local/bin/yq
    fi
fi

PROJECT_ID=$(yq '.gcp.project_id' $CONFIG_FILE)
BUCKET_NAME=$(yq '.gcp.bucket_name' $CONFIG_FILE)
ORCHESTRATOR_VM=$(yq '.orchestrator.vm_name' $CONFIG_FILE)
ZONE=$(yq '.gcp.zone' $CONFIG_FILE)

echo "🚀 Deploying $(yq '.project.name' $CONFIG_FILE) to $PROJECT_ID"

# 1. Git operations
echo "📝 Pushing latest changes..."
cd "$GIT_ROOT"

if ! git diff-index --quiet HEAD -- 2>/dev/null; then
    git add . > /dev/null 2>&1
    git commit -m "Deploy: $(date '+%Y-%m-%d %H:%M:%S')" > /dev/null 2>&1 || true
fi

git push origin main > /dev/null 2>&1 || git push origin master > /dev/null 2>&1 || {
    echo "❌ Failed to push to remote repository"
    exit 1
}

cd "$ORIGINAL_DIR"

# 2. Authenticate with GCP
echo "🔑 Authenticating with GCP..."

SERVICE_ACCOUNT_FILE="service-account.json"
if [ ! -f "$SERVICE_ACCOUNT_FILE" ]; then
    if [ -f "$GIT_ROOT/$SERVICE_ACCOUNT_FILE" ]; then
        SERVICE_ACCOUNT_FILE="$GIT_ROOT/$SERVICE_ACCOUNT_FILE"
    else
        echo "❌ service-account.json not found!"
        exit 1
    fi
fi

gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE" --quiet 2>/dev/null
gcloud config set project $PROJECT_ID --quiet 2>/dev/null

# 3. Upload configuration
echo "☁️ Uploading configuration..."
gsutil -q cp "$CONFIG_FILE" gs://$BUCKET_NAME/config/ 2>/dev/null

# 4. Process and upload .env file
echo "📄 Processing environment file..."

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
    # Get current orchestrator VM IP
    ORCHESTRATOR_IP=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")
    
    if [ -n "$ORCHESTRATOR_IP" ]; then
        # Create clean .env file for GCP
        cat > .env.gcp << GCP_ENV_EOF
# MLflow tracking server (set by deploy.sh)
MLFLOW_TRACKING_URI=http://$ORCHESTRATOR_IP:5000

GCP_ENV_EOF
        
        # Add other non-credential variables
        if [ -f "$ENV_FILE" ]; then
            grep -v "^MLFLOW_TRACKING_URI=" "$ENV_FILE" | \
            grep -v "^GOOGLE_APPLICATION_CREDENTIALS=" | \
            grep -v "^#" | \
            grep -v "^$" >> .env.gcp 2>/dev/null || true
        fi
        
        gsutil -q cp .env.gcp gs://$BUCKET_NAME/config/.env 2>/dev/null
        rm -f .env.gcp
        
        echo "✅ Environment configured with MLflow at $ORCHESTRATOR_IP:5000"
    else
        echo "⚠️ Could not get orchestrator IP - worker will determine MLflow URI"
    fi
else
    # Create basic .env file
    echo "MLFLOW_TRACKING_URI=http://orchestrator:5000" | gsutil -q cp - gs://$BUCKET_NAME/config/.env 2>/dev/null
fi

# 5. Upload data files
echo "📁 Checking data files..."
DATA_FILES_UPLOADED=0
DATA_FILES_SKIPPED=0

yq '.paths.data_files[]' "$CONFIG_FILE" | while read -r file; do
    # Find the data file
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
        
        # Check if file needs uploading
        if gsutil -q stat "$GCS_PATH" 2>/dev/null; then
            LOCAL_SIZE=$(stat -c%s "$DATA_FILE" 2>/dev/null || stat -f%z "$DATA_FILE" 2>/dev/null)
            REMOTE_SIZE=$(gsutil du "$GCS_PATH" 2>/dev/null | awk '{print $1}')
            
            if [ "$LOCAL_SIZE" = "$REMOTE_SIZE" ]; then
                continue  # Skip - same size
            fi
        fi
        
        # Upload the file
        gsutil -q cp "$DATA_FILE" "$GCS_PATH" 2>/dev/null
        echo "  📤 $(basename "$file")"
    fi
done

# 6. Check and start VM if needed
echo "🔍 Checking orchestrator VM..."
VM_STATUS=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" != "RUNNING" ]; then
    if [ "$VM_STATUS" = "NOT_FOUND" ]; then
        echo "❌ Orchestrator VM not found! Please create it first."
        exit 1
    else
        echo "🚀 Starting orchestrator VM..."
        gcloud compute instances start $ORCHESTRATOR_VM --zone=$ZONE > /dev/null 2>&1
        echo "⏳ Waiting for VM to be ready..."
        sleep 30
    fi
fi

# 7. Trigger deployment
echo "🎯 Triggering deployment..."
DEPLOY_ID=$(date '+%Y%m%d_%H%M%S')

gcloud compute instances add-metadata $ORCHESTRATOR_VM \
    --zone=$ZONE \
    --metadata deploy-trigger="$DEPLOY_ID" > /dev/null 2>&1

echo ""
echo "✅ Deployment triggered successfully!"
echo ""
echo "📊 Monitor: gcloud compute ssh $ORCHESTRATOR_VM --zone=$ZONE --command='sudo tmux attach-session -t orchestrator'"
echo "📈 MLflow: http://$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null):5000"
echo "📋 Logs: gcloud compute instances get-serial-port-output $ORCHESTRATOR_VM --zone=$ZONE"