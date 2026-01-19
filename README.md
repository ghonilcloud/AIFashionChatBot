# AI Fashion Design Generator

## Overview
AI Fashion Design Generator transforms fashion sketches into detailed designs using AI. Users upload a sketch, select design preferences through tones and Kansei words, and generate a styled fashion image.

## GitHub Repository
https://github.com/ghonilcloud/AIFashionChatBot

## Requirements
- Python 3.10 or higher
- Docker (for containerized deployment)
- Git
- NVIDIA GPU with CUDA support (recommended for local development)

## Environment Setup

### Step 1: Clone the Repository
```bash
git clone https://github.com/ghonilcloud/AIFashionChatBot.git
cd AIFashionChatBot
```

### Step 2: Create a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
HOST=0.0.0.0
PORT=8000
```

### Step 5: Download Required Models
```bash
python -c "from backend.app.image_generator import ImageGenerator; ImageGenerator()"
```

**Note:** Model downloads may take several minutes. The Stable Diffusion model is approximately 4-5 GB.

### Step 6: Verify Installation
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running the Application Locally

### Method 1: Using Python Script

1. **Start the Backend:**
   ```bash
   python start_server.py
   ```
   Backend API will be available at `http://localhost:8000`

2. **Open the Frontend:**
   ```bash
   cd frontend
   python -m http.server 3000
   ```
   Access at `http://localhost:3000`

### Method 2: Using Docker

1. **Build the Image:**
   ```bash
   docker build -t aifashionchatbot .
   ```

2. **Run the Container:**
   ```bash
   docker run -p 8000:8000 --gpus all aifashionchatbot
   ```

3. **Access:**
   - Backend: `http://localhost:8000`
   - Frontend: Open `frontend/index.html` in browser

### Troubleshooting

**PyTorch DLL Issues (Windows):**
```bash
.\fix_pytorch_dll.ps1
```

**CUDA Out of Memory:**
Reduce batch size or use CPU mode.

**Port Already in Use:**
Change port in `start_server.py` or set PORT environment variable.

## Cloud Deployment

### Deployment Architecture
- **Backend**: RunPod (GPU container hosting)
- **Frontend**: Vercel (Static site hosting)
- **CI/CD**: GitHub Actions
- **Container Registry**: GitHub Container Registry (GHCR)

### CI/CD Pipeline: GitHub Actions

#### Automated Docker Image Building
Every push to the `main` branch triggers GitHub Actions to build and push the Docker image to GHCR. RunPod automatically pulls the latest image.

#### Workflow Configuration
File: `.github/workflows/docker-build.yml`

```yaml
name: Build and Push Docker Image

on:
  push:
    branches: [ "main" ]

jobs:
  build:
    runs-on: ubuntu-latest

    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ghcr.io/${{ github.repository_owner }}/aifashionchatbot:latest
```

### Backend Deployment: RunPod

#### Step 1: Create Account
1. Go to https://runpod.io
2. Sign in with GitHub, Google, or email
3. Complete verification
4. Add payment method

#### Step 2: Navigate to Pods
1. Click "GPU Pods" in sidebar
2. Click "+ Deploy" button

#### Step 3: Select GPU

| **RTX 4090** | **24GB** | **Recommended** | **$0.34-0.69** |

#### Step 4: Configure Container
1. **Container Image:** `ghcr.io/ghonilcloud/aifashionchatbot:latest`
2. **Container Disk:** 20-50 GB
3. **Volume Disk:** 20-50 GB (optional, for model persistence)

#### Step 5: Network Configuration
1. **HTTP Ports:** Expose port `8000`
2. **TCP Ports:** Not required

#### Step 6: Environment Variables
```
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
HOST=0.0.0.0
PORT=8000
```

Get API keys:
- OpenAI: https://platform.openai.com/api-keys
- Hugging Face: https://huggingface.co/settings/tokens

#### Step 7: Deploy
1. Review configuration
2. Check estimated cost
3. Click "Deploy"
4. Wait 2-5 minutes for initialization

#### Step 8: Access Backend
1. Click "Connect" button on pod
2. Copy HTTPS URL (format: `https://[pod-id]-8000.proxy.runpod.net`)
3. Test at: `https://your-pod-url/docs`

#### Step 9: Configure Auto-Updates

**Option 1: Automatic**
- Enable "Auto-update container" in pod settings
- Set frequency (every 6-12 hours)

**Option 2: Manual**
- Stop pod → Start pod to pull latest image

### Frontend Deployment: Vercel

#### Step 1: Prepare Repository
Ensure `frontend` folder contains:
- `index.html`
- `script.js`
- `styles.css`

#### Step 2: Connect to Vercel
1. Go to https://vercel.com
2. Sign in with GitHub
3. Click "Add New Project"
4. Import AIFashionChatBot repository

#### Step 3: Configure Project
- **Framework Preset:** Other
- **Root Directory:** `frontend`
- **Build Command:** [empty]
- **Output Directory:** [default]
- **Install Command:** [empty]

#### Step 4: Configure Backend URL
Edit `frontend/script.js`:
```javascript
const API_BASE_URL = 'https://your-pod-id-8000.proxy.runpod.net';
```
Commit and push to GitHub.

#### Step 5: Deploy
1. Review configuration
2. Click "Deploy"
3. Wait 30-60 seconds
4. Access via provided Vercel URL

#### Step 6: Test Deployment
1. Open Vercel URL
2. Upload a sketch
3. Select preferences
4. Generate design to verify backend connection

#### Step 7: Configure CORS
Add Vercel URL to backend CORS in `backend/app/main.py`:
```python
allow_origins=[
    "http://localhost:3000",
    "https://your-project.vercel.app",
]
```
Commit and push changes.

### Monitoring

**RunPod:**
- GPU usage
- Cost tracking
- Application logs

**Vercel:**
- Analytics
- Deployment logs
- Performance metrics

**GitHub Actions:**
- Build status
- Workflow history

## User Guide

### How to Use

1. **Access the Application**
   - Local: `http://localhost:3000`
   - Production: Your Vercel URL

2. **Upload Sketch**
   - Click upload area or drag and drop
   - Supported: PNG, JPG, JPEG (max 10MB)
   - Preview appears after upload

3. **Select Design Preferences**
   - **Tones:** Select up to 3 from checkboxes
   - **Kansei Words:** Select aesthetic keywords
   - At least one selection required

4. **Generate Design**
   - Click "Generate Design" button
   - Wait 10-30 seconds
   - Generated design appears with original sketch

5. **View Results**
   - Compare original and generated design
   - View prompt and notes
   - Right-click image to download

### Troubleshooting

- **No response:** Check internet connection and backend status
- **Generation fails:** Verify GPU availability in backend logs
- **CORS errors:** Ensure frontend points to correct backend URL