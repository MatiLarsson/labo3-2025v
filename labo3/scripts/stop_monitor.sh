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