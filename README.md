# Nebula X Hackathon

Problem Statement 3: Predictive Fault Detection

Backend:

```
cd app/backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Frontend:

```
cd app/frontend
npm run dev
```

Then open http://localhost:5173.

Deployment:

Run when there are changes to frontend:
```
cd app/frontend
gcloud run deploy frontend --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars BACKEND_HOST=backend-7jonzrweja-uc.a.run.app
```

Run when there are changes to backend:
```
gcloud run deploy backend --source .
```