#!/bin/bash
# startup-script.sh - Simple orchestrator VM startup script with abort support

set -e
exec > >(tee -a /var/log/orchestrator.log) 2>&1

echo "$(date): 🚀 Orchestrator VM startup..."

# Install dependencies (gcloud is pre-installed on GCE)
apt-get update
apt-get install -y git curl tmux jq python3 python3-pip

# Install yq
if [ ! -f "/usr/local/bin/yq" ]; then
    wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
    chmod +x /usr/local/bin/yq
fi

# Setup workspace
WORKSPACE="/opt/orchestrator"
mkdir -p $WORKSPACE
cd $WORKSPACE

echo "$(date): 📋 VM uses service account attached to the instance for GCS access"
echo "$(date): 💡 No need to download service-account.json - using VM's default service account"

# Get bucket name from metadata
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket-name" -H "Metadata-Flavor: Google" 2>/dev/null || echo "labo3_bucket")
echo "$(date): 📦 Using bucket: $BUCKET_NAME"

# Create the main orchestrator script that will run in tmux
cat > /opt/orchestrator/main_orchestrator.sh << 'EOF'
#!/bin/bash
# Main orchestrator script - runs continuously in tmux with abort support

set -e

WORKSPACE="/opt/orchestrator"
cd $WORKSPACE

echo "🎯 Starting ML Orchestrator with abort support..."

# Get bucket name from VM metadata
BUCKET_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket-name" -H "Metadata-Flavor: Google" 2>/dev/null || echo "labo3_bucket")

# Function to download and load configuration
load_configuration() {
    echo "📋 Downloading configuration from gs://$BUCKET_NAME/config/project_config.yml"
    
    # Download latest configuration using VM's default service account
    if gsutil cp gs://$BUCKET_NAME/config/project_config.yml ./config.yml; then
        echo "✅ Configuration downloaded successfully"
    else
        echo "❌ Failed to download configuration from GCS"
        echo "💡 Make sure the VM's service account has access to gs://$BUCKET_NAME/config/"
        return 1
    fi
    
    # Parse configuration
    PROJECT_ID=$(yq '.gcp.project_id' ./config.yml)
    ZONE=$(yq '.gcp.zone' ./config.yml)
    CHECK_INTERVAL=$(yq '.orchestrator.check_interval' ./config.yml)
    
    # Set GCP project (VM should already be authenticated) - suppress confirmation prompts
    gcloud config set project $PROJECT_ID --quiet
    
    echo "📋 Project: $PROJECT_ID, Zone: $ZONE, Check Interval: ${CHECK_INTERVAL}s"
    
    # Export variables for use in functions
    export PROJECT_ID ZONE CHECK_INTERVAL BUCKET_NAME
}

# Initial configuration load
if ! load_configuration; then
    echo "❌ Failed to load initial configuration, exiting..."
    exit 1
fi

# Orchestration state
orchestration_active=false
current_job=0
total_jobs=0
completed_jobs=()  # Track completed jobs persistently

# Helper functions
get_job_name() {
    yq ".jobs[$1].name" ./config.yml
}

get_job_instance() {
    yq ".jobs[$1].instance_name" ./config.yml
}

get_job_script() {
    yq ".jobs[$1].script" ./config.yml
}

get_job_machine_type() {
    yq ".jobs[$1].machine_type" ./config.yml
}

get_job_dependencies() {
    yq ".jobs[$1].depends_on // \"\"" ./config.yml
}

check_instance_status() {
    local instance_name=$1
    gcloud compute instances describe $instance_name --zone=$ZONE --format="value(status)" 2>/dev/null || echo "NOT_FOUND"
}

check_job_completion() {
    local instance_name=$1
    local status=$(gcloud compute instances describe $instance_name --zone=$ZONE \
        --format="value(metadata.items[key=job-status].value)" 2>/dev/null || echo "")
    echo "${status:-running}"
}

# Record job completion persistently
record_job_completion() {
    local job_name=$1
    completed_jobs+=("$job_name")
    echo "📝 Recorded completion: $job_name"
    echo "✅ Completed jobs: ${completed_jobs[*]}"
}

# Improved dependency checking with persistent state
check_dependencies_completed() {
    local job_index=$1
    local dependencies=$(get_job_dependencies $job_index)
    
    if [ -z "$dependencies" ] || [ "$dependencies" = "null" ]; then
        return 0  # No dependencies
    fi
    
    echo "🔍 Checking dependencies for job $job_index: $dependencies"
    
    # Check if dependency job completed (using persistent record)
    for dep in $(echo "$dependencies" | tr ',' ' '); do
        local dep_completed=false
        
        # Check if dependency is in completed jobs array
        for completed_job in "${completed_jobs[@]}"; do
            if [ "$completed_job" = "$dep" ]; then
                dep_completed=true
                echo "✅ Dependency $dep already completed"
                break
            fi
        done
        
        # If not in completed array, check if it's currently running and completed
        if [ "$dep_completed" = false ]; then
            for i in $(seq 0 $((total_jobs-1))); do
                if [ "$(get_job_name $i)" = "$dep" ]; then
                    local dep_instance=$(get_job_instance $i)
                    local instance_status=$(check_instance_status $dep_instance)
                    
                    # Only check job completion if instance exists
                    if [ "$instance_status" != "NOT_FOUND" ]; then
                        local completion_status=$(check_job_completion $dep_instance)
                        if [ "$completion_status" = "completed" ]; then
                            dep_completed=true
                            echo "✅ Dependency $dep completed (live check)"
                            break
                        fi
                    fi
                fi
            done
        fi
        
        if [ "$dep_completed" = false ]; then
            echo "⏳ Dependency $dep not yet completed"
            return 1
        fi
    done
    
    echo "✅ All dependencies satisfied"
    return 0  # All dependencies completed
}

create_worker_startup_script() {
    local script_name=$1
    local instance_name=$2
    
    cat > /tmp/worker-startup.sh << WORKER_EOF
#!/bin/bash
set -e
exec > >(tee -a /var/log/worker.log) 2>&1

echo "\$(date): 🔧 Starting ML Worker..."
echo "📋 Script: $script_name"
echo "🏷️ Instance: $instance_name"

# Install dependencies (gcloud is pre-installed on GCE)
apt-get update
apt-get install -y curl git build-essential python3 python3-pip

# Install yq for YAML parsing
if [ ! -f "/usr/local/bin/yq" ]; then
    echo "📦 Installing yq..."
    wget -qO /usr/local/bin/yq https://github.com/mikefarah/yq/releases/latest/download/yq_linux_amd64
    chmod +x /usr/local/bin/yq
fi

# Setup workspace
mkdir -p /opt/worker
cd /opt/worker

# Set GCP project (gcloud is pre-installed on GCE instances)
gcloud config set project $PROJECT_ID --quiet

# Download configuration using VM's default service account
echo "📋 Downloading configuration..."
gsutil cp gs://$BUCKET_NAME/config/project_config.yml ./config.yml

# Clone/update repository for latest experiment code
echo "📦 Cloning/updating experiment repository..."
REPO_URL=\$(python3 -c "import yaml; print(yaml.safe_load(open('./config.yml'))['repository']['url'])")

# Clean clone approach
if [ -d "repo" ]; then
    rm -rf repo
fi

echo "📥 Cloning repository..."
git clone \$REPO_URL repo
cd repo

# Navigate to the labo3 subdirectory where the Python project is located
echo "📁 Navigating to labo3 subdirectory..."
if [ -d "labo3" ]; then
    cd labo3
    echo "✅ Found labo3 directory"
else
    echo "❌ labo3 directory not found in repository"
    echo "📂 Repository root contents:"
    ls -la
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
    exit 1
fi

# Verify we have pyproject.toml and project_config.yml
echo "📍 Current directory: \$(pwd)"
echo "📂 Contents:"
ls -la

if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found in labo3 directory"
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
    exit 1
fi

if [ ! -f "project_config.yml" ]; then
    echo "❌ project_config.yml not found in labo3 directory"
    echo "📁 Available files:"
    find . -name "*.yml" -o -name "*.yaml"
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
    exit 1
fi

# Install uv globally
echo "📦 Installing uv..."
pip3 install uv

# Create virtual environment using uv
echo "🔧 Creating virtual environment with uv..."
uv venv

# Install the project and all dependencies using uv
echo "📦 Installing project and dependencies in virtual environment..."
uv pip install -e .

echo "✅ Virtual environment created and project installed successfully"

# Download environment file (already cleaned by deploy.sh)
echo "📄 Loading environment..."
gsutil cp gs://$BUCKET_NAME/config/.env ./.env 2>/dev/null || echo "No .env file found"

# Get the current IP of the orchestrator VM (where MLflow is running)
echo "🔍 Getting orchestrator VM IP for MLflow..."
ORCHESTRATOR_VM_NAME=\$(yq '.orchestrator.vm_name' ./project_config.yml)
ORCHESTRATOR_IP=\$(gcloud compute instances describe \$ORCHESTRATOR_VM_NAME --zone=$ZONE --format="value(networkInterfaces[0].accessConfigs[0].natIP)" 2>/dev/null || echo "")

if [ -n "\$ORCHESTRATOR_IP" ]; then
    echo "📡 Found orchestrator IP: \$ORCHESTRATOR_IP"
    
    # Update or add MLflow URI to .env file
    if [ -f ".env" ]; then
        # Remove any existing MLFLOW_TRACKING_URI and add the current one
        grep -v "^MLFLOW_TRACKING_URI=" .env > .env.tmp || cp .env .env.tmp
        echo "MLFLOW_TRACKING_URI=http://\$ORCHESTRATOR_IP:5000" >> .env.tmp
        mv .env.tmp .env
        echo "✅ Updated MLflow URI to: http://\$ORCHESTRATOR_IP:5000"
    else
        # Create .env file with MLflow URI
        echo "MLFLOW_TRACKING_URI=http://\$ORCHESTRATOR_IP:5000" > .env
        echo "✅ Created .env with MLflow URI: http://\$ORCHESTRATOR_IP:5000"
    fi
else
    echo "⚠️ Could not get orchestrator IP"
    if [ ! -f ".env" ]; then
        echo "❌ No .env file and no orchestrator IP - cannot proceed"
        gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
        exit 1
    fi
fi

# Activate virtual environment and run the ML script
echo "🚀 Activating virtual environment and running ML script: $script_name"

# Source the virtual environment and run the script
source .venv/bin/activate

# Load environment variables after activating venv
set -a
[ -f ./.env ] && source ./.env
set +a

echo "🐍 Using Python: \$(which python)"
echo "🐍 Python version: \$(python --version)"
echo "📡 MLflow URI: \$MLFLOW_TRACKING_URI"

# Verify we have MLflow URI
if [ -z "\$MLFLOW_TRACKING_URI" ]; then
    echo "❌ MLFLOW_TRACKING_URI not set"
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
    exit 1
fi

# Data files will be downloaded by the ML script itself when needed
echo "💡 Data files will be downloaded by ML script when required"

# Execute the script from the scripts folder (we're already in labo3/)
if python scripts/$script_name 2>&1 | tee script_output.log; then
    SCRIPT_EXIT_CODE=0
    echo "✅ Script completed successfully"
else
    SCRIPT_EXIT_CODE=1
    echo "❌ Script failed with exit code \$SCRIPT_EXIT_CODE"
fi

# Upload results regardless of success/failure
echo "📤 Uploading results..."

# Get deployment ID safely (handle metadata access failures)
INSTANCE_NAME=\$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google" 2>/dev/null || echo "unknown-worker")
DEPLOY_ID=\$(date '+%Y%m%d_%H%M%S')  # Use timestamp as deployment ID
EXPERIMENT_NAME=\$(python -c "import yaml; print(yaml.safe_load(open('./project_config.yml'))['experiment_name'])" 2>/dev/null || echo "unknown")
TIMESTAMP=\$(date '+%Y%m%d_%H%M%S')

# Create organized results directory structure
RESULTS_PATH="results/\${EXPERIMENT_NAME}/\${DEPLOY_ID}/\${script_name}/\${instance_name}_\${TIMESTAMP}"
gsutil -m mkdir -p gs://$BUCKET_NAME/\$RESULTS_PATH/ 2>/dev/null || true

echo "📂 Uploading to: gs://$BUCKET_NAME/\$RESULTS_PATH/"

# Upload script logs with experiment context
if [ -f "script_output.log" ]; then
    echo "📋 Uploading script logs..."
    gsutil cp script_output.log gs://$BUCKET_NAME/\$RESULTS_PATH/script_output.log
fi

# Upload only ML-related files (exclude development files)
find . -maxdepth 2 \( -name "*.pkl" -o -name "*.joblib" -o -name "*.csv" -o -name "*.parquet" -o -name "*.json" -o -name "*.h5" -o -name "*.model" \) \\
    ! -path "./.vscode/*" ! -path "./.git/*" ! -path "./.*" | while read -r file; do
    if [ -f "\$file" ]; then
        echo "📤 Uploading: \$file"
        gsutil cp "\$file" gs://$BUCKET_NAME/\$RESULTS_PATH/\$(basename "\$file") || echo "⚠️ Failed to upload \$file"
    fi
done

# Signal completion status
if [ \$SCRIPT_EXIT_CODE -eq 0 ]; then
    echo "✅ Job completed successfully at \$(date)" | \\
        gsutil cp - gs://$BUCKET_NAME/status/\${EXPERIMENT_NAME}_\${DEPLOY_ID}_${instance_name}_success.txt
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=completed --zone=$ZONE
    echo "🎉 Worker completed successfully!"
else
    echo "❌ Job failed at \$(date)" | \\
        gsutil cp - gs://$BUCKET_NAME/status/\${EXPERIMENT_NAME}_\${DEPLOY_ID}_${instance_name}_failed.txt
    gcloud compute instances add-metadata \$(hostname) --metadata job-status=failed --zone=$ZONE
    echo "💥 Worker failed!"
fi

# Shutdown
echo "🛑 Shutting down instance..."
shutdown -h now
WORKER_EOF
}

create_worker_instance() {
    local job_index=$1
    local instance_name=$(get_job_instance $job_index)
    local script_name=$(get_job_script $job_index)
    local machine_type=$(get_job_machine_type $job_index)
    
    echo "🚀 Creating worker instance: $instance_name"
    
    # Create worker startup script
    create_worker_startup_script $script_name $instance_name
    
    # Create the instance with the same service account as orchestrator
    if gcloud compute instances create $instance_name \
        --zone=$ZONE \
        --machine-type=$machine_type \
        --image-family=ubuntu-2204-lts \
        --image-project=ubuntu-os-cloud \
        --scopes=https://www.googleapis.com/auth/cloud-platform \
        --preemptible \
        --metadata-from-file startup-script=/tmp/worker-startup.sh \
        --quiet; then
        
        echo "✅ Instance $instance_name created successfully"
        return 0
    else
        echo "❌ Failed to create instance $instance_name"
        return 1
    fi
}

delete_instance() {
    local instance_name=$1
    echo "🗑️  Deleting instance: $instance_name"
    gcloud compute instances delete $instance_name --zone=$ZONE --quiet 2>/dev/null || true
}

# Check for abort signal
check_abort_signal() {
    # Check metadata for abort trigger
    local abort_trigger=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/abort-trigger" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
    
    # Only proceed if we got a non-empty response that looks like a real trigger
    if [ -n "$abort_trigger" ] && [ "$abort_trigger" != "null" ] && [[ "$abort_trigger" != *"<!DOCTYPE"* ]] && [[ "$abort_trigger" != *"<html"* ]]; then
        echo "🛑 ABORT SIGNAL DETECTED: $abort_trigger"
        
        # Clear the abort trigger metadata
        INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
        gcloud compute instances remove-metadata $INSTANCE_NAME --keys=abort-trigger --zone=$ZONE --quiet
        
        # Also clear any pending deploy trigger
        gcloud compute instances remove-metadata $INSTANCE_NAME --keys=deploy-trigger --zone=$ZONE --quiet
        
        return 0
    fi
    
    return 1
}

# Abort current orchestration
abort_orchestration() {
    echo "🛑 ABORTING CURRENT ORCHESTRATION..."
    
    if [ "$orchestration_active" = true ]; then
        echo "🧹 Cleaning up all running job instances..."
        for i in $(seq 0 $((total_jobs-1))); do
            instance_name=$(get_job_instance $i)
            echo "🗑️ Terminating instance: $instance_name"
            delete_instance $instance_name
        done
        
        # Signal abort completion
        echo "$(date -Iseconds): Orchestration aborted by user" | gsutil cp - gs://$BUCKET_NAME/status/orchestration-aborted.txt
        
        echo "✅ All instances terminated"
    else
        echo "💡 No active orchestration to abort"
    fi
    
    # Reset orchestration state
    orchestration_active=false
    current_job=0
    total_jobs=0
    completed_jobs=()  # Reset completed jobs tracking
    
    echo "🔄 Returning to deployment signal monitoring..."
}

check_deployment_signal() {
    # Check metadata for deployment trigger
    local deploy_trigger=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/attributes/deploy-trigger" -H "Metadata-Flavor: Google" 2>/dev/null || echo "")
    
    # Only proceed if we got a non-empty response that looks like a real trigger (not HTML error)
    if [ -n "$deploy_trigger" ] && [ "$deploy_trigger" != "null" ] && [[ "$deploy_trigger" != *"<!DOCTYPE"* ]] && [[ "$deploy_trigger" != *"<html"* ]]; then
        echo "🎯 Deployment trigger detected: $deploy_trigger"
        
        # Re-download latest configuration when new deployment detected
        echo "🔄 Reloading configuration for new deployment..."
        if load_configuration; then
            echo "✅ Configuration reloaded successfully"
        else
            echo "❌ Failed to reload configuration, using existing config"
        fi
        
        # Clear the deployment trigger metadata
        INSTANCE_NAME=$(curl -s "http://metadata.google.internal/computeMetadata/v1/instance/name" -H "Metadata-Flavor: Google")
        gcloud compute instances remove-metadata $INSTANCE_NAME --keys=deploy-trigger --zone=$ZONE --quiet
        
        echo "🔄 Starting new orchestration cycle..."
        return 0
    fi
    
    return 1
}

start_orchestration() {
    echo "🚀 Starting new orchestration cycle..."
    orchestration_active=true
    current_job=0
    total_jobs=$(yq '.jobs | length' ./config.yml)
    completed_jobs=()  # Reset completed jobs tracking
    
    # Clean up any existing job instances
    echo "🧹 Cleaning up existing job instances..."
    for i in $(seq 0 $((total_jobs-1))); do
        instance_name=$(get_job_instance $i)
        delete_instance $instance_name
    done
    
    echo "📊 Starting orchestration with $total_jobs jobs"
}

# Main orchestration loop - this runs forever and actively manages workflow
echo "⏰ Starting orchestration monitor (checking every $CHECK_INTERVAL seconds)..."

while true; do
    echo "$(date): 🔄 Checking for signals..."
    
    # Check for abort signal first (highest priority)
    if check_abort_signal; then
        abort_orchestration
        # Continue to next iteration (skip deployment check)
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Check for new deployment signals
    if check_deployment_signal; then
        start_orchestration
    fi
    
    # Run job orchestration if active
    if [ "$orchestration_active" = true ]; then
        if [ $current_job -lt $total_jobs ]; then
            job_name=$(get_job_name $current_job)
            instance_name=$(get_job_instance $current_job)
            
            echo "🔍 Checking job: $job_name (instance: $instance_name) [$((current_job + 1))/$total_jobs]"
            
            # Check dependencies first
            if ! check_dependencies_completed $current_job; then
                echo "⏳ Dependencies not met for job $job_name, waiting..."
            else
                # Dependencies met, check instance status
                status=$(check_instance_status $instance_name)
                
                case $status in
                    "NOT_FOUND")
                        echo "📋 Instance not found, creating spot VM for $job_name..."
                        if create_worker_instance $current_job; then
                            echo "✅ Spot VM $instance_name created for $job_name"
                        else
                            echo "❌ Failed to create spot VM, will retry in next cycle"
                        fi
                        ;;
                        
                    "RUNNING")
                        echo "🟢 Instance $instance_name running, checking completion..."
                        completion=$(check_job_completion $instance_name)
                        case $completion in
                            "completed")
                                echo "✅ Job $job_name completed successfully"
                                
                                # Record completion BEFORE deleting instance
                                record_job_completion "$job_name"
                                
                                echo "🗑️ Cleaning up completed instance..."
                                delete_instance $instance_name
                                ((current_job++))
                                echo "📈 Moving to next job ($current_job/$total_jobs)"
                                
                                # If there's a next job, immediately trigger it
                                if [ $current_job -lt $total_jobs ]; then
                                    next_job_name=$(get_job_name $current_job)
                                    echo "🚀 Ready to start next job: $next_job_name"
                                fi
                                ;;
                            "failed")
                                echo "❌ Job $job_name failed! Recreating spot VM..."
                                delete_instance $instance_name
                                echo "⏰ Waiting 60 seconds before recreating failed job..."
                                sleep 60
                                ;;
                            *)
                                echo "🔄 Job $job_name still running on $instance_name..."
                                ;;
                        esac
                        ;;
                        
                    "TERMINATED"|"STOPPING"|"STOPPED")
                        echo "🔍 Instance $instance_name stopped, checking if job completed first..."
                        completion=$(check_job_completion $instance_name)
                        if [ "$completion" = "completed" ]; then
                            echo "✅ Job $job_name completed successfully before shutdown"
                            
                            # Record completion BEFORE deleting instance
                            record_job_completion "$job_name"
                            
                            delete_instance $instance_name
                            ((current_job++))
                            echo "📈 Moving to next job ($current_job/$total_jobs)"
                        elif [ "$completion" = "failed" ]; then
                            echo "❌ Job $job_name failed before shutdown"
                            delete_instance $instance_name
                            echo "⏰ Waiting 60 seconds before recreating failed job..."
                            sleep 60
                        else
                            echo "🔄 Job $job_name incomplete - instance stopped unexpectedly, recreating..."
                            delete_instance $instance_name
                            echo "⏰ Waiting 30 seconds before recreating stopped instance..."
                            sleep 30
                        fi
                        ;;
                        
                    "PROVISIONING"|"STAGING")
                        echo "🟡 Instance $instance_name starting up (status: $status)..."
                        ;;
                        
                    *)
                        echo "🟡 Instance $instance_name in unexpected state: $status, monitoring..."
                        ;;
                esac
            fi
        else
            # All jobs completed successfully
            echo "🎉 All jobs in workflow completed successfully!"
            
            # Final cleanup
            echo "🧹 Final cleanup of any remaining instances..."
            for i in $(seq 0 $((total_jobs-1))); do
                instance_name=$(get_job_instance $i)
                delete_instance $instance_name
            done
            
            # Signal completion
            echo "📝 Signaling orchestration completion..."
            echo "$(date -Iseconds): All jobs completed successfully" | gsutil cp - gs://$BUCKET_NAME/status/orchestration-complete.txt
            
            echo "✅ Workflow orchestration cycle finished! Waiting for new deployments..."
            orchestration_active=false
            current_job=0
            completed_jobs=()  # Reset for next deployment
        fi
    else
        echo "😴 No active orchestration, waiting for deployment trigger..."
    fi
    
    echo "⏰ Sleeping for $CHECK_INTERVAL seconds before next check..."
    sleep $CHECK_INTERVAL
done
EOF

chmod +x /opt/orchestrator/main_orchestrator.sh

# Kill existing tmux session if it exists
tmux kill-session -t orchestrator 2>/dev/null || true

# Start the main orchestrator in tmux - this will run forever
echo "$(date): 📺 Starting persistent orchestrator in tmux session..."
tmux new-session -d -s orchestrator "/opt/orchestrator/main_orchestrator.sh"

echo "$(date): 🎉 Startup complete!"
echo "$(date): 💡 To view orchestrator logs: sudo tmux attach-session -t orchestrator"
echo "$(date): 💡 To abort experiments: use abort.sh script from local machine"