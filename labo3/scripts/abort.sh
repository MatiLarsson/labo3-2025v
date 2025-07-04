#!/bin/bash
# abort.sh - Abort current ML experiment pipeline
# Run this from your local machine to stop the current experiment

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
ORCHESTRATOR_VM=$(yq '.orchestrator.vm_name' $CONFIG_FILE)
ZONE=$(yq '.gcp.zone' $CONFIG_FILE)

echo "🛑 ABORTING EXPERIMENT PIPELINE"
echo "📋 Project: $PROJECT_ID"
echo "🖥️  Orchestrator VM: $ORCHESTRATOR_VM"
echo "🌍 Zone: $ZONE"
echo ""

# Authenticate with service account
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

# Check if VM is running
echo "🔍 Checking orchestrator VM status..."
VM_STATUS=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" = "NOT_FOUND" ]; then
    echo "❌ Orchestrator VM not found! Cannot abort."
    exit 1
elif [ "$VM_STATUS" != "RUNNING" ]; then
    echo "❌ Orchestrator VM is not running (status: $VM_STATUS). Cannot abort."
    exit 1
fi

echo "✅ Orchestrator VM is running"

# Confirm abort action
echo ""
echo "⚠️  WARNING: This will:"
echo "   • Stop the current ML experiment pipeline"
echo "   • Terminate all running worker instances"
echo "   • Reset orchestration to idle state"
echo "   • Keep the orchestrator VM running and listening for new deployments"
echo ""
read -p "Are you sure you want to abort the current experiment? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🚫 Abort cancelled."
    exit 0
fi

# Send abort signal
echo "🛑 Sending abort signal to orchestrator..."
ABORT_ID=$(date '+%Y%m%d_%H%M%S')

gcloud compute instances add-metadata $ORCHESTRATOR_VM \
    --zone=$ZONE \
    --metadata abort-trigger="$ABORT_ID"

echo "✅ Abort signal sent with ID: $ABORT_ID"
echo ""
echo "⏰ The orchestrator will:"
echo "   1. Detect the abort signal within ~60 seconds"
echo "   2. Terminate all running worker instances"
echo "   3. Reset to idle state"
echo "   4. Continue listening for new deployment triggers"
echo ""
echo "📊 Monitor the abort process with:"
echo "  gcloud compute ssh $ORCHESTRATOR_VM --zone=$ZONE --command='tmux attach-session -t orchestrator'"
echo ""
echo "📋 Check orchestrator logs:"
echo "  gcloud compute instances get-serial-port-output $ORCHESTRATOR_VM --zone=$ZONE"
echo ""
echo "🎯 To start a new experiment after abort:"
echo "  ./scripts/deploy.sh"