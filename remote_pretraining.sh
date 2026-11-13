#!/bin/bash
set -e

QUEUED_RESOURCE_ID="vl-1x1-queued"
ZONE="us-west4-a"
RUN_TIMESTAMP=$(date +"%Y-%m%d_%H-%M%S")
LOG_PATH="gs://toy-gpt/training-logs/${RUN_TIMESTAMP}.log"

echo "Uploading current local code state to GCS..."

gcloud storage cp pretrain.py hellaswag.py pyproject.toml uv.lock "gs://toy-gpt/code/"

echo "Local files successfully synced, creating the startup script."

cat << EOF > /tmp/vl_1x1_startup_script.sh
#!/bin/bash
set -e
gcloud storage cp gs://toy-gpt/code/* .
curl -LsSf https://astral.sh/uv/install.sh | sh
source \$HOME/.local/bin/env
uv venv
source .venv/bin/activate
uv python install 3.12
uv sync --python 3.12
uv run pretrain.py --data_root=gs://toy-gpt/edu_fineweb10B --hellaswag_dir=gs://toy-gpt/hellaswag > training.log 2>&1 || true
gcloud storage cp training.log ${LOG_PATH}
gcloud compute tpus tpu-vm delete node-1 --zone=us-west4-a --quiet
EOF

echo "Startup script created, requesting the tpu."

gcloud compute tpus queued-resources delete "${QUEUED_RESOURCE_ID}" \
    --zone="${ZONE}" \
    --quiet --force --async 2>/dev/null || true

echo "Waiting for old queued resource to be fully deleted..."
while gcloud compute tpus queued-resources describe "${QUEUED_RESOURCE_ID}" --zone="${ZONE}" &>/dev/null; do
    sleep 5
done

gcloud compute tpus queued-resources create "${QUEUED_RESOURCE_ID}" \
	--node-id="${RUN_TIMESTAMP}" \
    --zone="${ZONE}" \
    --accelerator-type="v5litepod-1" \
    --runtime-version="v2-alpha-tpuv5-lite" \
    --metadata-from-file="startup-script=/tmp/vl_1x1_startup_script.sh"

echo "Queued resource submitted successfully!"

while true; do
    # Check current state of the TPU VM
    echo "Checking state."
    STATE=$(gcloud compute tpus tpu-vm describe "${RUN_TIMESTAMP}" --zone="${ZONE}" --format="get(state)" 2>/dev/null)
    
    echo "Got ${STATE}."
    if [[ "${STATE}" == "READY" ]]; then
        echo "==> TPU VM is provisioned and training is running."
        break
    elif [[ -n "${STATE}" ]]; then
        echo "Current TPU VM state: ${STATE}. Waiting..."
    else
        echo "TPU VM not yet created in queue. Waiting..."
    fi
    
    sleep 30
done

while true; do
    # Check if the underlying TPU node still exists
    if ! gcloud compute tpus tpu-vm describe "${RUN_TIMESTAMP}" --zone="${ZONE}" &>/dev/null; then
        echo "==> TPU VM has finished and terminated."
        break
    fi
    sleep 60
done

# Clean up the queued resource wrapper
gcloud compute tpus queued-resources delete "${QUEUED_RESOURCE_ID}" \
    --zone="${ZONE}" \
    --quiet 2>/dev/null || true

echo "Workflow finished and the VM was cleaned up. Logs can be inspected in ${LOG_PATH}"
