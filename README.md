# Nebula X Hackathon

## Problem Statement 3: Predictive Fault Detection

Tasked to predict **four independent subsystems** of a rail vehicle and develop a user-friendly interface for engineers and operational users to interpret the results.

## Features

- Introduction page: briefly explains how to use the app and the 4 subsystems.
- Predict page: form-like uploads (drag-and-drop and multi-file supported) for each subsystem. Detailing what is required in the input files for each subsystem and form validation to ensure inputs match the requirements for each subsystem.
- History page: history of past records submitted along with navigation to dashboard pages.
- Dashboard page: for insights and interpretability of predictions for each subsystem.

## Tech Stack

![ReactJS](https://img.shields.io/badge/-ReactJs-61DAFB?logo=react&logoColor=white&style=for-the-badge)
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Supabase](https://img.shields.io/badge/Supabase-3ECF8E?style=for-the-badge&logo=supabase&logoColor=white)
![Google Cloud Run](https://img.shields.io/badge/Cloud_Run-Deployed-4285F4?style=for-the-badge&logo=googlecloud&logoColor=white)

We used React for our frontend, FastAPI for our backend, Supabase for record and file storage. Our app is deployed on Google Cloud Run and can be accessed through https://frontend-205373376635.us-central1.run.app/

## Model Performance

[TODO]

## Get started

**Frontend:**

```
cd app/frontend
npm run dev
```

**Backend:**

```
cd app/backend
source .venv/bin/activate
uvicorn main:app --reload --port 8000
```

Then open http://localhost:5173.

**Deployment:**

Run when there are updates to frontend:
```
cd app/frontend
gcloud run deploy frontend --source . --region us-central1 --allow-unauthenticated \
  --set-env-vars BACKEND_HOST=backend-7jonzrweja-uc.a.run.app
```

Run when there are updates to backend:
```
cd app/backend
gcloud run deploy backend --source .
```

## Contributors

| Name                        | GitHub Username   |
|-----------------------------|-------------------|
| Ian Buxton                  | [Buxt-Codes](https://github.com/Buxt-Codes)
| Rachel Tan                  | [Racheltmz](https://github.com/Racheltmz)
| Tan Yichen                  | [sultanyichen](https://github.com/sultanyichen)

