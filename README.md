AI Recommender System - Setup Instructions

This project implements a modular, domain-agnostic content-based recommender with TF–IDF and LLM embedding pipelines, exposed via a FastAPI backend and wire-up to a Voiceflow conversational front-end via ngrok.

Prerequisites:
- Python 3.13.1 (or compatible 3.10+)
- Git (to clone this repo)
- VS Code (or your preferred editor/IDE)
- ngrok (for exposing your local FastAPI server to Voiceflow)
- Voiceflow account (free plan is sufficient)

Local Setup:
1. Clone the repo:

2. Create & activate a virtual environment:
   # Unix / macOS
   python3 -m venv venv
   source venv/bin/activate

   # Windows (PowerShell)
   python -m venv venv
   .\\venv\\Scripts\\Activate.ps1

3. Install dependencies:
   pip install -r requirements.txt


Running the API + ngrok:
1. Start your FastAPI server:
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload

2. Expose it with ngrok:
   ngrok http 8000
   - Copy the HTTPS “Forwarding” URL (e.g. https://abcd1234.ngrok.io)

3. Paste into Voiceflow:
   - Open your Voiceflow project.
   - In each API/Webhook step, update the Base URL to your current ngrok URL + endpoint path.
   - Note: ngrok URLs rotate on restart; update your Voiceflow steps each time.

Voiceflow Setup:
1. Create a free account at https://www.voiceflow.com
2. Import the project:
   - Go to Projects → Import and upload voiceflow/flow.json
3. Configure API steps:
   - For each “Recommend” block, set the request URL to: <ngrok-url>/recommend_tfidf_user or similar
   - Ensure HTTP method is POST and JSON body matches the FastAPI schema.

Repository Structure:
.
├── app.py                  # FastAPI application & endpoints
├── pipeline1_test.py       # Offline tester for P-AUTO
├── tfidf_test.py           # Offline tester for P-TF
├── requirements.txt        # Python dependencies
├── voiceflow/
│   └── flow.json           # Exported Voiceflow project
└── README.txt              # This file

Testing & Validation:
- Offline tests:
  python pipeline1_test.py
  python tfidf_test.py
- Logging:
  All API calls recorded in recommendations.log


