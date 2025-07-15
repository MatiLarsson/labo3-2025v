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

# Kill ALL existing monitoring processes and sessions on node0
echo "🗑️ Cleaning up previous monitoring processes..."

# Kill monitor script processes
if gcloud compute ssh node0 --zone=us-east1-d --command="sudo pkill -f monitor_script.sh" 2>/dev/null; then
    echo "✅ Killed existing monitor script processes"
else
    echo "ℹ️ No monitor scripts to kill"
fi

# Kill sudo tmux monitor sessions
if gcloud compute ssh node0 --zone=us-east1-d --command="sudo tmux kill-session -t monitor" 2>/dev/null; then
    echo "✅ Killed sudo tmux monitor session"
else
    echo "ℹ️ No sudo tmux monitor session to kill"
fi

echo "🎯 Cleanup completed"

# Push code
echo "📦 Pushing code to repository..."
git add -A && git commit -m "Deploy $(date)" 2>/dev/null || echo "No changes to commit"
git push --set-upstream origin main 2>/dev/null || git push 2>/dev/null || echo "⚠️ Git push failed, continuing anyway"

# Auth GCP
if [ -f "service-account.json" ]; then
    echo "Service account file found, activating service account..."
    gcloud auth activate-service-account --key-file=service-account.json --quiet 2>/dev/null
else
    echo "No service account file found, using default credentials..."
fi

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

gsutil cp .env.gcp gs://$BUCKET_NAME/config/.env 2>/dev/null || gsutil cp .env.gcp gs://$BUCKET_NAME/config/.env
rm -f .env.gcp
echo "✅ Environment configured with MLflow at $NODE0_IP:5000"

# Upload data files (only if they don't exist)
yq -r '.paths.data_files[]' $CONFIG_FILE | while IFS= read -r file; do
    if [ -n "$file" ] && [ "$file" != "null" ]; then
        if ! gsutil -q stat gs://$BUCKET_NAME/$file 2>/dev/null; then
            if [ -f "$file" ]; then
                gsutil cp "$file" gs://$BUCKET_NAME/$file 2>/dev/null && echo "📤 Uploaded: $file" || echo "⚠️ Upload failed: $file"
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
WORKER_EXISTS=$(gcloud compute instances list --format="value(name,zone)" --filter="name:$INSTANCE_NAME" 2>/dev/null || echo "")

if [ ! -z "$WORKER_EXISTS" ]; then
    # Extract the zone from the result (format is "name zone")
    EXISTING_ZONE=$(echo "$WORKER_EXISTS" | awk '{print $2}')
    EXISTING_NAME=$(echo "$WORKER_EXISTS" | awk '{print $1}')
    
    echo "🗑️ Deleting existing instance '$EXISTING_NAME' in zone $EXISTING_ZONE..."

    gcloud compute instances delete $EXISTING_NAME --zone=$EXISTING_ZONE --quiet 2>/dev/null || echo "⚠️ Could not delete $EXISTING_NAME instance"
    
    # Wait for worker instance to be fully deleted
    echo "⏳ Waiting for instance '$EXISTING_NAME' to be fully deleted..."
    while true; do
        sleep 10

        WORKER_STATUS=$(gcloud compute instances list --format="value(name)" --filter="name:$INSTANCE_NAME" 2>/dev/null || echo "")
        
        if [ -z "$WORKER_STATUS" ]; then
            echo "✅ Instance '$INSTANCE_NAME' deleted successfully"
            break
        else
            echo "⏳ Still waiting for '$INSTANCE_NAME' deletion..."
        fi
    done
else
    echo "✅ No existing instance named '$INSTANCE_NAME' found"
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

# Start the ML script
tmux send-keys -t ml "python scripts/$SCRIPT_NAME 2>&1 | tee run.log; echo \$? > /tmp/ml_exit_code; echo ML_SCRIPT_DONE > /tmp/ml_done" Enter

# Clean up the background preemption monitor
kill $MONITOR_PID 2>/dev/null || true

# Disable trap to avoid cleanup on exit
trap - SIGTERM SIGINT SIGQUIT

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

# Create completion status file with retry
echo "Job completed at $(date) with exit code $EXIT_CODE" > /tmp/completion_status.txt

# Upload with retry logic
UPLOAD_SUCCESS=false
for i in {1..3}; do
    echo "📤 Upload attempt $i/3..."
    if gsutil cp /tmp/completion_status.txt gs://$BUCKET_NAME/run_logs/completed_${DEPLOY_ID}.txt 2>/dev/null; then
        echo "✅ Completion status uploaded successfully"
        UPLOAD_SUCCESS=true
        break
    else
        echo "⚠️ Upload attempt $i failed, retrying in 5 seconds..."
        sleep 5
    fi
done

if [ "$UPLOAD_SUCCESS" = false ]; then
    echo "❌ Failed to upload completion status after 3 attempts"
fi

INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
INSTANCE_ZONE=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/zone" -H "Metadata-Flavor: Google" | sed 's|.*/||')

# Erase vm
gcloud compute instances delete $INSTANCE_NAME --zone=$INSTANCE_ZONE --quiet || echo "⚠️ Could not delete instance"
EOF

# Create instance with zone fallback
echo "🖥️ Creating instance with zone fallback..."

# Extract region from worker zone
REGION=$(echo $WORKER_ZONE | sed 's/-[a-z]$//')
CURRENT_ZONE_SUFFIX=$(echo $WORKER_ZONE | sed 's/.*-//')

# Define zone rotation: b->c->d->b...
get_next_zone() {
    local current_suffix="$1"
    case $current_suffix in
        "b") echo "c" ;;
        "c") echo "d" ;;
        "d") echo "b" ;;
        *) echo "b" ;;  # Default fallback
    esac
}

INSTANCE_CREATED=false
ATTEMPT=1
MAX_ATTEMPTS=10  # Prevent infinite loops
CURRENT_ZONE=$WORKER_ZONE

while [ $ATTEMPT -le $MAX_ATTEMPTS ] && [ "$INSTANCE_CREATED" = false ]; do
    echo "🎯 Attempt $ATTEMPT: Trying zone $CURRENT_ZONE..."
    
    # Create the instance (disable exit on error for this command)
    set +e

    CREATE_OUTPUT=$(gcloud compute instances create $INSTANCE_NAME \
        --zone=$CURRENT_ZONE \
        --machine-type=$MACHINE_TYPE \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --boot-disk-size=$BOOT_DISK_SIZE \
        --boot-disk-type=$BOOT_DISK_TYPE \
        --scopes=cloud-platform \
        --preemptible \
        --metadata-from-file startup-script=/tmp/startup.sh \
        --metadata project-id=$PROJECT_ID,bucket-name=$BUCKET_NAME,script-name=$SCRIPT_NAME,repo-url=$REPO_URL \
        2>/dev/null)
    
    CREATE_EXIT_CODE=$?

    set -e  # Re-enable exit on error
    
    if [ $CREATE_EXIT_CODE -eq 0 ]; then
        echo "✅ Instance created successfully in zone $CURRENT_ZONE"
        WORKER_ZONE=$CURRENT_ZONE
        INSTANCE_CREATED=true
        break
    else
        echo "❌ Zone $CURRENT_ZONE unavailable, trying next zone..."
        
        # Get next zone in rotation
        CURRENT_ZONE_SUFFIX=$(echo $CURRENT_ZONE | sed 's/.*-//')
        NEXT_ZONE_SUFFIX=$(get_next_zone $CURRENT_ZONE_SUFFIX)
        CURRENT_ZONE="${REGION}-${NEXT_ZONE_SUFFIX}"
        
        echo "🔄 Trying next zone: $CURRENT_ZONE"
        sleep 5
    fi
    
    ATTEMPT=$((ATTEMPT + 1))
done

if [ "$INSTANCE_CREATED" = true ]; then
    # Filter out warnings from output
    echo "$CREATE_OUTPUT" | grep -v "WARNING:" | grep -v "Some requests generated warnings:" | grep -v "Disk size.*is larger than image size" | grep -v "You might need to resize"
    
    # Erase previous logs
    echo "🧹 Cleaning up previous logs..."
    gsutil -m rm -r gs://$BUCKET_NAME/run_logs/ 2>/dev/null || echo "📂 No previous results to clean"

    # Copy startup script to node0 for daemon use (overwrite if exists)
    echo "📤 Copying startup script to node0..."
    gcloud compute scp /tmp/startup.sh node0:/tmp/startup.sh --zone=$NODE0_ZONE --quiet

    cat > /tmp/monitor_config.env << CONFIG_EOF
PROJECT_ID=$PROJECT_ID
BUCKET_NAME=$BUCKET_NAME
INSTANCE_NAME=$INSTANCE_NAME
WORKER_ZONE=$WORKER_ZONE
MACHINE_TYPE=$MACHINE_TYPE
BOOT_DISK_SIZE=$BOOT_DISK_SIZE
BOOT_DISK_TYPE=$BOOT_DISK_TYPE
SCRIPT_NAME=$SCRIPT_NAME
REPO_URL=$REPO_URL
NODE0_ZONE=$NODE0_ZONE
CONFIG_EOF

    # Create monitoring script
    cat > /tmp/monitor_script.sh << 'MONITOR_SCRIPT_EOF'
#!/bin/bash
source /tmp/monitor_config.env

export JOB_NAME="monitor_ml_job_worker"
echo "🤖 Starting ML job monitoring daemon at $(date)"
echo "📊 Monitoring bucket: gs://$BUCKET_NAME/run_logs/"

# Clean up any persistent log files from previous monitor runs
echo "🧹 Cleaning up previous monitor logs..."
rm -f /tmp/monitor_complete_history.log
echo "✅ Previous monitor logs cleaned"

while true; do
    echo "🔍 $(date): Checking job status..."
    
    if gsutil -q stat gs://$BUCKET_NAME/run_logs/completed_*.txt 2>/dev/null; then
        echo "✅ $(date): Job completed successfully - stopping daemon"
        break
    fi

    # Check worker instance status every cycle
    WORKER_STATUS=$(gcloud compute instances describe $INSTANCE_NAME --zone=$WORKER_ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
    echo "📊 $(date): Worker status: $WORKER_STATUS"
    
    if [ "$WORKER_STATUS" = "TERMINATED" ] || [ "$WORKER_STATUS" = "STOPPED" ] || [ "$WORKER_STATUS" = "NOT_FOUND" ]; then
        echo "🚨 $(date): Worker was preempted/stopped! Triggering restart..."
        echo "🗑️ $(date): Force deleting worker instance..."
        gcloud compute instances delete $INSTANCE_NAME --zone=$WORKER_ZONE --quiet 2>/dev/null || echo "Instance already deleted"
        echo "⏳ $(date): Waiting for worker instance deletion..."
        while gcloud compute instances list --filter="name:$INSTANCE_NAME" --format="value(name)" | grep -q "$INSTANCE_NAME"; do
            echo "⏳ $(date): Still waiting for $INSTANCE_NAME deletion..."
            sleep 30
        done
        
        echo "🔄 $(date): Recreating worker instance..."
        RECREATE_SUCCESS=false
        RECREATE_ATTEMPT=1
        
        while [ $RECREATE_ATTEMPT -le 60 ] && [ "$RECREATE_SUCCESS" = false ]; do
            echo "🎯 $(date): Recreate attempt $RECREATE_ATTEMPT in zone $WORKER_ZONE..."
            
            if gcloud compute instances create $INSTANCE_NAME \
                --zone=$WORKER_ZONE \
                --machine-type=$MACHINE_TYPE \
                --image-family=ubuntu-2204-lts \
                --image-project=ubuntu-os-cloud \
                --boot-disk-size=$BOOT_DISK_SIZE \
                --boot-disk-type=$BOOT_DISK_TYPE \
                --scopes=cloud-platform \
                --preemptible \
                --metadata-from-file startup-script=/tmp/startup.sh \
                --metadata project-id=$PROJECT_ID,bucket-name=$BUCKET_NAME,script-name=$SCRIPT_NAME,repo-url=$REPO_URL 2>/dev/null; then
                
                echo "✅ $(date): Instance recreated successfully"
                RECREATE_SUCCESS=true
                
                echo "Instance $INSTANCE_NAME recreated at $(date)" > /tmp/recreated_status.txt
                gsutil cp /tmp/recreated_status.txt gs://$BUCKET_NAME/run_logs/recreated_$(date +%Y%m%d_%H%M%S).txt 2>/dev/null
                rm -f /tmp/recreated_status.txt
            else
                echo "❌ $(date): Instance creation failed, retrying in 60 seconds..."
                sleep 60
                RECREATE_ATTEMPT=$((RECREATE_ATTEMPT + 1))
            fi
        done
        
        if [ "$RECREATE_SUCCESS" = false ]; then
            echo "💥 $(date): Failed to recreate instance after 60 attempts"
        fi
    fi
    
    echo "📤 $(date): Uploading monitor logs..."
    # Use a persistent log file that accumulates history
    PERSISTENT_LOG="/tmp/monitor_complete_history.log"

    # Append current tmux session with timestamp
    echo "=== Monitor Session Capture at $(date) ===" >> "$PERSISTENT_LOG"
    tmux capture-pane -t monitor -p >> "$PERSISTENT_LOG" 2>/dev/null || echo "Monitor session log at $(date)" >> "$PERSISTENT_LOG"
    echo "" >> "$PERSISTENT_LOG"  # Add separator

    # Upload the complete accumulated history
    gsutil cp "$PERSISTENT_LOG" gs://$BUCKET_NAME/run_logs/monitor.log 2>/dev/null || echo "Could not upload monitor logs"

    echo "💤 $(date): Sleeping for 5 minutes..."
    sleep 300
done

echo "🏁 $(date): Monitoring daemon finished"
MONITOR_SCRIPT_EOF

    # Create simple daemon script
    cat > /tmp/monitor_daemon.sh << 'DAEMON_SCRIPT_EOF'
#!/bin/bash
if ! command -v tmux &> /dev/null; then
    echo "📦 Installing tmux on node0..."
    sudo apt-get update -qq && sudo apt-get install -y tmux
fi

# Make sure past monitor processes are killed
echo "🗑️  Making sure no past monitoring sessions or processes are left running..."

if sudo pkill -f monitor_script.sh 2>/dev/null; then
    echo "Killed existing monitor script processes"
else
    echo "No past monitor script processes to kill"
fi

if sudo tmux kill-session -t monitor 2>/dev/null; then
    echo "Killed existing monitor session"
else
    echo "No existing monitor session to kill"
fi

source /tmp/monitor_config.env
gcloud config set project $PROJECT_ID --quiet >/dev/null 2>&1

sudo tmux new-session -d -s monitor
sudo tmux send-keys -t monitor 'chmod +x /tmp/monitor_script.sh && /tmp/monitor_script.sh' Enter

echo "🤖 Monitoring daemon started in tmux session 'monitor'"
echo "📊 View logs: gcloud compute ssh node0 --zone=$NODE0_ZONE --command='sudo tmux attach -t monitor'"
DAEMON_SCRIPT_EOF

    # Copy all files to node0
    echo "📤 Copying monitoring files to node0..."
    gcloud compute scp /tmp/monitor_config.env node0:/tmp/monitor_config.env --zone=$NODE0_ZONE --quiet
    gcloud compute scp /tmp/monitor_script.sh node0:/tmp/monitor_script.sh --zone=$NODE0_ZONE --quiet
    gcloud compute scp /tmp/monitor_daemon.sh node0:/tmp/monitor_daemon.sh --zone=$NODE0_ZONE --quiet
    
    echo "🤖 Starting monitoring daemon on node0..."
    gcloud compute ssh node0 --zone=$NODE0_ZONE --command="chmod +x /tmp/monitor_daemon.sh /tmp/monitor_script.sh && /tmp/monitor_daemon.sh"
    
    # Monitor instances
    echo "📊 Monitor: gcloud compute ssh $INSTANCE_NAME --zone=$WORKER_ZONE --command='sudo tmux attach -t ml'"
    echo "📁 All logs will be uploaded to gs://$BUCKET_NAME/run_logs/ even if preempted"
else
    echo "❌ Failed to create instance after $((ATTEMPT-1)) attempts"
    echo "🛑 Last attempted zone: $CURRENT_ZONE"
    exit 1
fi

rm -f /tmp/startup.sh
rm -f /tmp/monitor_daemon.sh
rm -f /tmp/monitor_script.sh
rm -f /tmp/monitor_config.env