# Radiology AI Backend

This is the backend service for the Radiology AI application, which provides AI-powered analysis of chest X-ray images using the CheXagent model.

## Setup and Deployment

### Local Development

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the development server:
```bash
python main.py
```

### Deployment to Render

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the service:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python main.py`
   - Environment Variables:
     - `PORT`: 8000 (or let Render set it automatically)

## API Endpoints

### POST /api/generate
Generates AI analysis for chest X-ray images.

Request body:
```json
{
  "paths": ["base64_encoded_image_data"],
  "prompt": "Your prompt here"
}
```

Response:
```json
{
  "response": "AI-generated analysis"
}
```

### GET /health
Health check endpoint.

Response:
```json
{
  "status": "healthy"
}
```

## Environment Variables

- `PORT`: The port number for the server (default: 8000) 