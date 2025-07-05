#!/bin/bash
# abort.sh - Abort current ML experiment pipeline
# Run this from your local machine to stop the current experiment

set -e

# Find git repository root first
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
        echo "❌ project_config.yml not found in any expected location!"
        exit 1
    fi
fi

# Parse YAML config (requires yq)
if ! command -v yq &> /dev/null; then
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

echo "🛑 Aborting experiment: $ORCHESTRATOR_VM ($PROJECT_ID)"

# Authenticate with service account
SERVICE_ACCOUNT_FILE="service-account.json"
if [ ! -f "$SERVICE_ACCOUNT_FILE" ]; then
    if [ -f "$GIT_ROOT/$SERVICE_ACCOUNT_FILE" ]; then
        SERVICE_ACCOUNT_FILE="$GIT_ROOT/$SERVICE_ACCOUNT_FILE"
    elif [ -f "$GIT_ROOT/labo3/$SERVICE_ACCOUNT_FILE" ]; then
        SERVICE_ACCOUNT_FILE="$GIT_ROOT/labo3/$SERVICE_ACCOUNT_FILE"
    elif [ -f "labo3/$SERVICE_ACCOUNT_FILE" ]; then
        SERVICE_ACCOUNT_FILE="labo3/$SERVICE_ACCOUNT_FILE"
    else
        echo "❌ service-account.json not found in any expected location!"
        exit 1
    fi
fi

gcloud auth activate-service-account --key-file="$SERVICE_ACCOUNT_FILE" --quiet
gcloud config set project $PROJECT_ID --quiet

# Check if VM is running
VM_STATUS=$(gcloud compute instances describe $ORCHESTRATOR_VM --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")

if [ "$VM_STATUS" = "NOT_FOUND" ]; then
    echo "❌ Orchestrator VM not found! Cannot abort."
    exit 1
elif [ "$VM_STATUS" != "RUNNING" ]; then
    echo "❌ Orchestrator VM is not running (status: $VM_STATUS). Cannot abort."
    exit 1
fi

# Confirm abort action
echo "⚠️  This will stop the current experiment and terminate all worker instances."
read -p "Continue? (y/N): " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "🚫 Aborted."
    exit 0
fi

# Send abort signal
ABORT_ID=$(date '+%Y%m%d_%H%M%S')

gcloud compute instances add-metadata $ORCHESTRATOR_VM \
    --zone=$ZONE \
    --metadata abort-trigger="$ABORT_ID"

echo "✅ Abort signal sent. Workers will terminate within ~60 seconds."
echo "📊 Monitor: gcloud compute ssh $ORCHESTRATOR_VM --zone=$ZONE --command='tmux attach-session -t orchestrator'"