#!/bin/bash

echo "🛑 Stopping all monitoring processes..."

# Kill monitor script processes on node0 (always in us-east1-d)
if gcloud compute ssh node0 --zone=us-east1-d --command="sudo pkill -f monitor_script.sh" 2>/dev/null; then
    echo "✅ Killed existing monitor script processes"
else
    echo "ℹ️ No monitor scripts to kill"
fi

# Kill tmux monitor sessions on node0 (always in us-east1-d)
if gcloud compute ssh node0 --zone=us-east1-d --command="sudo tmux kill-session -t monitor" 2>/dev/null; then
    echo "✅ Killed tmux monitor session"
else
    echo "ℹ️ No tmux monitor session to kill"
fi

# Find the worker instance in any us-east1 zone
echo "🔍 Looking for worker instance in us-east1..."
INSTANCE_INFO=$(gcloud compute instances list --filter="name=worker AND zone:(us-east1-b OR us-east1-c OR us-east1-d) AND status=RUNNING" --format="value(name,zone)" 2>/dev/null)

if [ -n "$INSTANCE_INFO" ]; then
    INSTANCE_NAME=$(echo "$INSTANCE_INFO" | cut -f1)
    INSTANCE_ZONE=$(echo "$INSTANCE_INFO" | cut -f2)
    echo "📍 Found running instance: $INSTANCE_NAME in zone $INSTANCE_ZONE"
    
    # Kill the worker instance
    echo "🗑️ Terminating worker instance..."
    if gcloud compute instances delete $INSTANCE_NAME --zone=$INSTANCE_ZONE --quiet 2>/dev/null; then
        echo "✅ Worker instance terminated"
    else
        echo "⚠️ Failed to terminate worker instance"
    fi
else
    echo "ℹ️ No running worker instance found in us-east1"
fi

echo "🏁 Abort complete"