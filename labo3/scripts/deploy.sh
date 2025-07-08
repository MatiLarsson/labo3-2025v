#!/bin/bash
# deploy.sh - Simple ML job deployment with preemption handling

set -e

# Find config file
CONFIG_FILE="project_config.yml"
[ ! -f "$CONFIG_FILE" ] && CONFIG_FILE="$(git rev-parse --show-toplevel)/labo3/project_config.yml"

# Parse config
PROJECT_ID=$(yq '.gcp.project_id' $CONFIG_FILE)
BUCKET_NAME=$(yq '.gcp.bucket_name' $CONFIG_FILE)
NODE0_ZONE=$(yq '.gcp.node0_zone' $CONFIG_FILE)
WORKER_ZONE=$(yq '.gcp.worker_zone' $CONFIG_FILE)
SCRIPT_NAME=$(yq '.jobs.script' $CONFIG_FILE)
INSTANCE_NAME=$(yq '.jobs.instance_name' $CONFIG_FILE)
MACHINE_TYPE=$(yq '.jobs.machine_type' $CONFIG_FILE)
REPO_URL=$(yq '.repository.url' $CONFIG_FILE)

# Parse disk configuration
BOOT_DISK_SIZE=$(yq '.jobs.boot_disk_size // "100GB"' $CONFIG_FILE)
BOOT_DISK_TYPE=$(yq '.jobs.boot_disk_type // "pd-standard"' $CONFIG_FILE)

echo "🚀 Deploying ML job: $INSTANCE_NAME"

# Push code
git add -A && git commit -m "Deploy $(date)" || true
git push --set-upstream origin main 2>/dev/null || git push 2>/dev/null || echo "⚠️ Git push failed, continuing anyway"

# Auth GCP
gcloud auth activate-service-account --key-file=service-account.json --quiet 2>/dev/null
gcloud config set project $PROJECT_ID --quiet 2>/dev/null

# Process and upload .env file
echo "📄 Processing environment file..."
ENV_FILE=""
if [ -f ".env" ]; then
    ENV_FILE=".env"
elif [ -f "$(git rev-parse --show-toplevel)/.env" ]; then
    ENV_FILE="$(git rev-parse --show-toplevel)/.env"
elif [ -f "$(git rev-parse --show-toplevel)/labo3/.env" ]; then
    ENV_FILE="$(git rev-parse --show-toplevel)/labo3/.env"
fi

# Overwrite .env with settings needed for GCP vms
NODE0_IP=$(gcloud compute instances describe node0 --zone=$NODE0_ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)")
cat > .env.gcp << GCP_ENV_EOF
# MLflow tracking server (set by deploy.sh)
MLFLOW_TRACKING_URI=http://$NODE0_IP:5000
GCP_ENV_EOF

if [ -f "$ENV_FILE" ]; then
    grep -v "^MLFLOW_TRACKING_URI=" "$ENV_FILE" | \
    grep -v "^GOOGLE_APPLICATION_CREDENTIALS=" | \
    grep -v "^#" | \
    grep -v "^$" >> .env.gcp 2>/dev/null || true
fi

gsutil cp .env.gcp gs://$BUCKET_NAME/config/.env
rm -f .env.gcp
echo "✅ Environment configured with MLflow at $NODE0_IP:5000"

# Upload data files (only if they don't exist)
yq -r '.paths.data_files[]' $CONFIG_FILE | while IFS= read -r file; do
    if [ -n "$file" ] && [ "$file" != "null" ]; then
        if ! gsutil -q stat gs://$BUCKET_NAME/$file 2>/dev/null; then
            if [ -f "$file" ]; then
                gsutil cp "$file" gs://$BUCKET_NAME/$file && echo "📤 Uploaded: $file"
            else
                echo "⚠️ File not found: $file"
            fi
        else
            echo "✅ Exists: $file"
        fi
    fi
done

# Check for existing worker instance and clean it up
echo "🔍 Checking for existing instance named '$INSTANCE_NAME'..."
WORKER_EXISTS=$(gcloud compute instances list --zones=$WORKER_ZONE --format="value(name)" --filter="name:$INSTANCE_NAME" 2>/dev/null || echo "")

if [ ! -z "$WORKER_EXISTS" ]; then
    echo "🗑️ Deleting existing instance in zone $WORKER_ZONE: $WORKER_EXISTS"

    gcloud compute instances delete $INSTANCE_NAME --zone=$WORKER_ZONE --quiet 2>/dev/null || echo "⚠️ Could not delete $INSTANCE_NAME instance"
    
    # Wait for worker instance to be fully deleted
    echo "⏳ Waiting for instance '$INSTANCE_NAME' to be fully deleted..."
    while true; do
        sleep 10

        WORKER_STATUS=$(gcloud compute instances list --zones=$WORKER_ZONE --format="value(name)" --filter="name:$INSTANCE_NAME" 2>/dev/null || echo "")
        
        if [ -z "$WORKER_STATUS" ]; then
            echo "✅ Instance '$INSTANCE_NAME' deleted successfully"
            break
        else
            echo "⏳ Still waiting for '$INSTANCE_NAME' deletion..."
        fi
    done
else
    echo "✅ No existing instance named '$INSTANCE_NAME' found in zone $WORKER_ZONE"
fi

# Create startup script with preemption handling
cat > /tmp/startup.sh << 'EOF'
#!/bin/bash
set -e

echo "🔧 Expanding filesystem to use full disk..."
# Expand the filesystem to use the full disk size
sudo resize2fs /dev/sda1 2>/dev/null || echo "⚠️ Resize2fs not needed or failed"

# Check available disk space
echo "💾 Disk space after expansion:"
df -h /

apt-get update && apt-get install -y git tmux python3-pip python3-venv wget
wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
chmod +x /usr/local/bin/yq
pip3 install uv

cd /opt

PROJECT_ID=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/project-id" -H "Metadata-Flavor: Google")
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket-name" -H "Metadata-Flavor: Google")
SCRIPT_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/script-name" -H "Metadata-Flavor: Google")
REPO_URL=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/repo-url" -H "Metadata-Flavor: Google")

gcloud config set project $PROJECT_ID --quiet

# Create cleanup function that uploads logs
cleanup_and_upload() {
    echo "🚨 Cleanup triggered - uploading logs before shutdown..."
    
    cd /opt/repo/labo3 2>/dev/null || cd /opt
    
    DEPLOY_ID=$(date '+%Y%m%d_%H%M%S')
    INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
    
    # Upload run.log if it exists
    if [ -f "run.log" ]; then
        gsutil cp run.log gs://$BUCKET_NAME/run_logs/run_${DEPLOY_ID}_interrupted.log 2>/dev/null || echo "⚠️ Could not upload run.log"
        echo "✅ Uploaded interrupted run log"
    fi
    
    # Upload any other relevant files
    if [ -f "error.log" ]; then
        gsutil cp error.log gs://$BUCKET_NAME/run_logs/error_${DEPLOY_ID}.log 2>/dev/null || echo "⚠️ Could not upload error.log"
    fi
    
    # Create a status file indicating the job was interrupted
    echo "Job interrupted at $(date) on instance $INSTANCE_NAME" > /tmp/interrupted_status.txt
    gsutil cp /tmp/interrupted_status.txt gs://$BUCKET_NAME/run_logs/interrupted_${DEPLOY_ID}.txt 2>/dev/null || echo "⚠️ Could not upload status"
    
    echo "🔄 Cleanup completed"
    exit 0
}

# Set up preemption monitoring in background
monitor_preemption() {
    while true; do
        # Exit if main script completed successfully
        if [ -f /tmp/ml_done ]; then
            echo "🎯 Main script completed - stopping preemption monitor"
            exit 0
        fi
        
        # Check preemption notice every 5 seconds
        if curl -s "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google" | grep -q "TRUE"; then
            echo "⚠️ PREEMPTION NOTICE RECEIVED - Starting cleanup..."
            cleanup_and_upload
            break
        fi
        sleep 5
    done
} &

# Store the background process PID so we can clean it up later
MONITOR_PID=$!

# Also set up signal handlers for other shutdown scenarios
trap cleanup_and_upload SIGTERM SIGINT SIGQUIT

echo "🛡️ Preemption monitoring started"

echo "Starting ML script in tmux..."
tmux new-session -d -s ml || echo "⚠️ Failed to start tmux"
tmux send-keys -t ml "git clone $REPO_URL repo && cd repo/labo3" Enter
tmux send-keys -t ml "gsutil cp gs://$BUCKET_NAME/config/.env .env 2>/dev/null || echo 'No .env file found'" Enter
tmux send-keys -t ml "echo 'Contents of .env file:' && cat .env" Enter
tmux send-keys -t ml "python3 -m venv .venv && source .venv/bin/activate" Enter
tmux send-keys -t ml "export \$(cat .env | xargs) && echo 'MLFLOW_TRACKING_URI=' \$MLFLOW_TRACKING_URI" Enter
tmux send-keys -t ml "uv sync" Enter
tmux send-keys -t ml "echo '💾 Final disk check before ML script:' && df -h /" Enter

# Start the ML script with better error handling
tmux send-keys -t ml "python scripts/$SCRIPT_NAME 2>&1 | tee run.log; echo \$? > /tmp/ml_exit_code" Enter
tmux send-keys -t ml "echo ML_SCRIPT_DONE > /tmp/ml_done" Enter

echo "Waiting for ML script to complete..."
while [ ! -f /tmp/ml_done ]; do 
    # Check if we're being preempted while waiting
    if curl -s "http://metadata.google.internal/computeMetadata/v1/instance/preempted" -H "Metadata-Flavor: Google" | grep -q "TRUE"; then
        echo "⚠️ PREEMPTION DURING EXECUTION - Starting cleanup..."
        cleanup_and_upload
        exit 0
    fi
    
    echo "Still waiting... $(date)"
    sleep 30
done

echo "ML script completed, uploading results..."

# Clean up the background preemption monitor
kill $MONITOR_PID 2>/dev/null || true

cd /opt/repo/labo3

DEPLOY_ID=$(date '+%Y%m%d_%H%M%S')

# Check exit code
EXIT_CODE=$(cat /tmp/ml_exit_code 2>/dev/null || echo "unknown")
if [ "$EXIT_CODE" = "0" ]; then
    LOG_SUFFIX="success"
else
    LOG_SUFFIX="error_${EXIT_CODE}"
fi

# Upload run.log
gsutil cp run.log gs://$BUCKET_NAME/run_logs/run_${DEPLOY_ID}_${LOG_SUFFIX}.log 2>/dev/null || echo "⚠️ Could not upload run.log"

# Create completion status file
echo "Job completed at $(date) with exit code $EXIT_CODE" > /tmp/completion_status.txt
gsutil cp /tmp/completion_status.txt gs://$BUCKET_NAME/run_logs/completed_${DEPLOY_ID}.txt 2>/dev/null || echo "⚠️ Could not upload completion status"

INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
INSTANCE_ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | sed 's|.*/||')

# Erase vm
gcloud compute instances delete $INSTANCE_NAME --zone=$INSTANCE_ZONE --quiet || echo "⚠️ Could not delete instance"
EOF

# Create instance
echo "🖥️ Creating instance"
gcloud compute instances create $INSTANCE_NAME \
    --zone=$WORKER_ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=ubuntu-2204-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=$BOOT_DISK_SIZE \
    --boot-disk-type=$BOOT_DISK_TYPE \
    --scopes=cloud-platform \
    --preemptible \
    --metadata-from-file startup-script=/tmp/startup.sh \
    --metadata project-id=$PROJECT_ID,bucket-name=$BUCKET_NAME,script-name=$SCRIPT_NAME,repo-url=$REPO_URL \
    2>&1 | grep -v "WARNING:" | grep -v "Some requests generated warnings:" | grep -v "Disk size.*is larger than image size" | grep -v "You might need to resize"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Instance created successfully"
    echo "📊 Monitor: gcloud compute ssh $INSTANCE_NAME --zone=$WORKER_ZONE --command='sudo tmux attach -t ml'"
    echo "📁 Logs will be uploaded to gs://$BUCKET_NAME/run_logs/ even if preempted"
    # Erase previous logs
    echo "🧹 Cleaning up previous logs..."
    gsutil -m rm -r gs://$BUCKET_NAME/run_logs/ 2>/dev/null || echo "📂 No previous results to clean"
else
    echo "❌ Instance creation failed"
    exit 1
fi

rm /tmp/startup.sh