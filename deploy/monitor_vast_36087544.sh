#!/bin/bash
set -u
INSTANCE_ID=36087544
PROJECT=/Users/revanthgundala/projects/omx-training
LOCAL_OUT=$PROJECT/outputs
STATE_DIR=$PROJECT/outputs/.vast_monitor
mkdir -p "$STATE_DIR"
LOG=$STATE_DIR/monitor_${INSTANCE_ID}.log

export VASTAI_API_KEY=bb986b1a1f0435e14ec92256f23c38c947e7def342ca6288a3fb13decd27cc5e

cd "$PROJECT"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

log "monitor started for instance $INSTANCE_ID"

while true; do
  STATUS_LOGS=$(uv run python -c "
from vastai_sdk import VastAI
v = VastAI(api_key='$VASTAI_API_KEY')
try:
    inst = v.show_instance(id=$INSTANCE_ID)
    print('STATUS::', inst.get('actual_status'), '::', inst.get('status_msg', '')[:100])
except Exception as e:
    print('STATUS::ERROR::', e)
try:
    logs = v.logs(instance_id=$INSTANCE_ID, tail='80')
    print('LOGS::')
    print(logs if isinstance(logs, str) else str(logs))
except Exception as e:
    print('LOGS_ERR::', e)
" 2>&1)

  echo "$STATUS_LOGS" | grep -E '^STATUS::' | tee -a "$LOG"

  # Check for actual training failure first (OOM, CUDA error, traceback)
  if echo "$STATUS_LOGS" | grep -qE 'OutOfMemoryError|CUDA error|RuntimeError|=== REMOTE FAILED|Traceback \(most recent'; then
    log "❌ training failure detected (OOM/error/traceback)"
    log "leaving instance up for inspection. NOT destroying."
    exit 1
  fi
  if echo "$STATUS_LOGS" | grep -q 'TRAINING COMPLETE'; then
    log "✅ TRAINING COMPLETE detected"
    break
  fi
  if echo "$STATUS_LOGS" | grep -qE 'STATUS::(offline|exited|error|dead)'; then
    log "⚠️  bad instance status; aborting (NOT destroying)"
    exit 1
  fi

  sleep 120
done

log "fetching scp URL..."
SCP_URL=$(uv run python -c "
from vastai_sdk import VastAI
v = VastAI(api_key='$VASTAI_API_KEY')
print(v.scp_url(id=$INSTANCE_ID))
" 2>&1)
log "scp_url: $SCP_URL"

# scp_url returns like scp://root@ssh8.vast.ai:16276
SSH_HOST=$(echo "$SCP_URL" | sed -E 's#scp://[^@]+@([^:]+):.*#\1#')
SSH_PORT=$(echo "$SCP_URL" | sed -E 's#.*:([0-9]+).*#\1#')
log "host=$SSH_HOST port=$SSH_PORT"

# safety: verify destination dir doesn't already exist
if [ -d "$LOCAL_OUT/pi05_pour_water_relative" ]; then
  log "⚠️  $LOCAL_OUT/pi05_pour_water_relative already exists; renaming with timestamp"
  mv "$LOCAL_OUT/pi05_pour_water_relative" "$LOCAL_OUT/pi05_pour_water_relative.bak_$(date +%s)"
fi

log "scp'ing /workspace/outputs/pi05_pour_water_relative -> $LOCAL_OUT/"
scp -o StrictHostKeyChecking=no -P "$SSH_PORT" -r \
    "root@$SSH_HOST:/workspace/outputs/pi05_pour_water_relative" \
    "$LOCAL_OUT/" 2>&1 | tee -a "$LOG"

if [ ! -d "$LOCAL_OUT/pi05_pour_water_relative" ]; then
  log "❌ SCP failed; NOT destroying instance"
  exit 1
fi
log "✅ checkpoints copied to $LOCAL_OUT/pi05_pour_water_relative"

# Verify HF push happened (lerobot-train pushes via --policy.repo_id)
log "verifying HF repo RevanthGundala/pi05-pour-water-relative-3k exists..."
HF_OK=$(uv run python -c "
from huggingface_hub import HfApi
try:
    info = HfApi().model_info('RevanthGundala/pi05-pour-water-relative-3k')
    print('HF_OK')
except Exception as e:
    print('HF_MISSING:', e)
" 2>&1)
log "HF check: $HF_OK"

if echo "$HF_OK" | grep -q HF_MISSING; then
  log "⚠️  HF repo missing; uploading manually..."
  uv run python -c "
from huggingface_hub import HfApi
api = HfApi()
api.create_repo('RevanthGundala/pi05-pour-water-relative-3k', private=False, exist_ok=True)
api.upload_folder(folder_path='$LOCAL_OUT/pi05_pour_water_relative/checkpoints/last/pretrained_model', repo_id='RevanthGundala/pi05-pour-water-relative-3k')
print('uploaded')
" 2>&1 | tee -a "$LOG"
fi

log "destroying instance $INSTANCE_ID..."
uv run python -c "
from vastai_sdk import VastAI
v = VastAI(api_key='$VASTAI_API_KEY')
v.destroy_instance(id=$INSTANCE_ID)
print('destroyed')
" 2>&1 | tee -a "$LOG"

log "🎉 done"
