#!/bin/bash
set -e

USER="rafiki53"
ZONE="us-west4-a"
RUN_TIMESTAMP=$(date +"%Y-%m%d_%H-%M%S")
QUEUED_RESOURCE_ID="vl-1x1-queued-${RUN_TIMESTAMP}"
LOG_PATH="gs://toy-gpt/training-logs/${RUN_TIMESTAMP}.log"
TRACE_PATH="gs://toy-gpt/training-traces/${RUN_TIMESTAMP}"

echo "Uploading current local code state to GCS..."

gcloud storage cp pretrain.py hellaswag.py pyproject.toml uv.lock "gs://toy-gpt/code/"

echo "Local files successfully synced, creating the startup script."

cat << EOF > /tmp/vl_1x1_startup_script.sh
#!/bin/bash
set -e
DIR="/home/rafiki53/app"
mkdir -p "\$DIR"
cd "\$DIR"
gcloud storage cp gs://toy-gpt/code/* .
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.local/bin/env
uv venv --python 3.12
source .venv/bin/activate
uv python install 3.12
uv sync --python 3.12
export PYTHONUNBUFFERED=1
(
  while true; do
    sleep 30
    gcloud storage cp training.log ${LOG_PATH} --quiet 2>/dev/null || true
  done
) &
SYNC_PID=\$!
uv run pretrain.py --data_root=gs://toy-gpt/edu_fineweb10B > training.log 2>&1 || true
kill \$SYNC_PID 2>/dev/null || true
gcloud storage cp training.log ${LOG_PATH}
gcloud compute tpus tpu-vm delete ${RUN_TIMESTAMP} --zone=us-west4-a --quiet
EOF

echo "Startup script created, requesting the tpu."

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
    STATE=$(gcloud compute tpus tpu-vm describe "${RUN_TIMESTAMP}" --zone="${ZONE}" --format="get(state)" 2>/dev/null || echo "PROVISIONING_OR_QUEUED")
    
    echo "Got ${STATE}."
    if [[ "${STATE}" == "READY" ]]; then
        echo "==> TPU VM is provisioned and training is running. To view the logs run: gcloud storage cat ${LOG_PATH} | less"
        break
    elif [[ -n "${STATE}" ]]; then
        echo "Current TPU VM state: ${STATE}. Waiting..."
    else
        echo "TPU VM not yet created in queue. Waiting..."
    fi
    
    sleep 30
done
