# Google Cloud GPU Training Setup

## Quick Start (Run these commands)

### Step 1: Install Google Cloud SDK
```powershell
# Download and install
Invoke-WebRequest -Uri "https://dl.google.com/dl/cloudsdk/channels/rapid/GoogleCloudSDKInstaller.exe" -OutFile "$env:TEMP\gcloud_installer.exe"
& "$env:TEMP\gcloud_installer.exe"
```

### Step 2: Initialize and Login
```cmd
# Open NEW terminal after installation
C:\Google\Cloud SDK\bin\gcloud.cmd init
C:\Google\Cloud SDK\bin\gcloud.cmd auth login
```

### Step 3: Set Project
```cmd
gcloud config set project YOUR_PROJECT_ID
gcloud services enable compute.googleapis.com
```

### Step 4: Request GPU Quota (IMPORTANT!)
1. Go to: https://console.cloud.google.com/iam-admin/quotas
2. Search for "NVIDIA T4 GPUs"
3. Click "Edit Quotas"
4. Request 1-2 GPUs in us-central1
5. Wait for approval (usually instant for new accounts)

### Step 5: Create GPU VM
```bash
gcloud compute instances create trm-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=100GB \
    --boot-disk-type=pd-ssd \
    --metadata="install-nvidia-driver=True"
```

### Step 6: Upload Your Files
```bash
# In your Dexter-Eternal directory:
tar -czf training_files.tar.gz train_memory_trm.py train_tool_trm.py train_reasoning_trm.py train_all_trms.py maximum_trm_config.py requirements.txt

gcloud compute scp training_files.tar.gz trm-training-vm:~ --zone=us-central1-a
gcloud compute scp -r dexter_TRMs trm-training-vm:~ --zone=us-central1-a
gcloud compute scp -r core trm-training-vm:~ --zone=us-central1-a
gcloud compute scp -r TinyRecursiveModels trm-training-vm:~ --zone=us-central1-a
```

### Step 7: Start Training
```bash
gcloud compute ssh trm-training-vm --zone=us-central1-a

# On the VM:
tar -xzf training_files.tar.gz
mkdir -p dexter_TRMs/models
pip install torch torchvision tqdm

# Start all 3 in parallel
python train_memory_trm.py > memory.log 2>&1 &
python train_tool_trm.py > tool.log 2>&1 &
python train_reasoning_trm.py > reasoning.log 2>&1 &

# Monitor progress
tail -f memory.log
tail -f tool.log
tail -f reasoning.log
```

### Step 8: Download Results
```bash
# When training is done, download checkpoints:
gcloud compute scp --recurse trm-training-vm:~/dexter_TRMs/models dexter_TRMs/ --zone=us-central1-a
```

### Step 9: Stop VM (SAVE MONEY!)
```bash
gcloud compute instances stop trm-training-vm --zone=us-central1-a
```

## Costs
- **T4 GPU**: ~$0.35/hour
- **n1-standard-8**: ~$0.38/hour
- **Total**: ~$0.73/hour while training
- **Est. training time**: 2-4 hours for all 3 TRMs
- **Total cost**: ~$2-3

## Alternative: Use Preemptible (Spot) VMs
Save 70% cost (but VM may be interrupted):
```bash
gcloud compute instances create trm-training-vm \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator=type=nvidia-tesla-t4,count=1 \
    --preemptible \
    ... (rest same)
```

## Monitoring Training
```bash
# SSH into VM
gcloud compute ssh trm-training-vm --zone=us-central1-a

# Check GPU usage
nvidia-smi

# Check logs
tail -f memory.log
tail -f tool.log
tail -f reasoning.log
```

## Troubleshooting

### GPU not found
```bash
# Check if NVIDIA driver installed
nvidia-smi

# If not, restart VM or install manually
gcloud compute instances stop trm-training-vm --zone=us-central1-a
gcloud compute instances start trm-training-vm --zone=us-central1-a
```

### Out of memory
Reduce batch size in the scripts (change to batch_size=1)

### Quota denied
Request GPU quota at console.cloud.google.com/iam-admin/quotas

## One-Command Setup Script
```powershell
# Run this in PowerShell as Administrator
irm https://raw.githubusercontent.com/googlecloudplatform/cloud-sdk/master/install.ps1 | iex
```
