# EcoRoute AI — Complete Step-by-Step Guide (commands + screenshots)

Everything is in order. Each phase lists: **what you do → exact commands → what you should see → 📸 screenshot to take**.
Screenshot IDs (SS-01, SS-02 …) are the proof list for your report. Suggested report section is shown in brackets.

> Commands are written for **Windows PowerShell inside the VS Code terminal** (`Ctrl + ` `). Mac/Linux differences are noted.
> `gcloud` commands are deliberately on **one line** so they work in PowerShell, CMD and Mac/Linux.

---

## PHASE 0 — Install the tools (once)

| Tool | Version | Check command |
|---|---|---|
| Python | **3.11** (use the same as the Docker image) | `python --version` |
| VS Code | latest | – |
| Git | latest | `git --version` |
| Docker Desktop | latest (must be *running*) | `docker --version` |
| Google Cloud CLI (gcloud) | latest | `gcloud --version` |

📸 **SS-01** — terminal showing the four version commands. *(Report: Environment / Tools)*

---

## PHASE 1 — Create the project in VS Code

1. Unzip `EcoRouteAI.zip`, then in VS Code: **File → Open Folder → EcoRouteAI**.
2. Open the terminal (`Ctrl + ``) and create a virtual environment:

```powershell
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. Confirm the dataset is at `data/energydata_complete.csv` (it is already copied from your zip).

📸 **SS-02** — VS Code Explorer showing the folder structure (`api, data, models, monitoring, src, tests, Dockerfile …`). *(Report: Project structure)*

**Why a virtual environment?** It isolates the project's library versions. The model `.pkl` files are version-sensitive, so the versions in `requirements.txt` must be the same when you train, test, build Docker and deploy.

---

## PHASE 2 — Data preprocessing and baseline models (Module 1: AI lifecycle)

### 2.1 Inspect the data and split it chronologically
```powershell
python -m src.preprocessing
```
You will see: shape (19735 × 29), date range, missing values (0), target statistics, and the **70/15/15 chronological split** with date ranges.

📸 **SS-03** — the whole output (inspection + split table). *(Report: Dataset and preprocessing)*

### 2.2 Justify the feature set (no test-set peeking)
```powershell
python -m src.feature_selection_cv
```
This compares three feature sets using forward-in-time cross-validation on train+validation only. In my run, the set that included absolute temperatures was clearly **worse** on future data. The reason: the data runs from winter to late spring, so temperature acts like a calendar proxy and the model learns "winter patterns" that fail in spring. That is why `config.py` uses humidity, lighting, wind, visibility and time features.

📸 **SS-04** — the 3-row table. *(Report: Feature engineering / seasonal shift)*

### 2.3 Train the Decision Tree and Random Forest
```powershell
python -m src.train_models
```
It tunes small grids on the **validation** set, saves `models/decision_tree.pkl`, `models/random_forest.pkl`, the drift baseline `models/drift_baseline.json`, and prints test-set **MAE, RMSE, latency** for both models.

📸 **SS-05** — tuning lines + the final baseline table. *(Report: Baseline model results)*
📸 **SS-06** — VS Code Explorer showing the new files inside `models/`.

---

## PHASE 3 — Router and the main experiment

### 3.1 Train the router
```powershell
python -m src.train_router
```
Shows validation MAE of DT and RF, the **margin search table** (how many Wh better RF must be before it is worth using) and the selected margin.

📸 **SS-07** — router output + margin search table. *(Report: Routing methodology)*

### 3.2 Compare DT only vs RF only vs EcoRoute
```powershell
python -m src.evaluate
```
Creates `results/final_comparison.csv` and `results/comparison_charts.png`.

📸 **SS-08** — terminal table. **SS-09** — open `results/comparison_charts.png`. *(Report: Results — your most important table)*

> **Be honest in the report.** The three approaches will have close accuracy because the dataset is noisy. The finding to report is: *"routing sent X % of requests to RF, kept accuracy at least as good as the individual models in our test period, and reduced average latency/compute compared with RF-only."* One test period is not a statistical proof, so say "in our experiment".

---

## PHASE 4 — Run and test the API locally (Module 2: model serving)

```powershell
uvicorn api.main:app --port 8080
```
Open **http://localhost:8080/docs** in the browser.

1. `GET /health` → `{"status":"healthy"}`
2. `POST /predict` → click *Try it out* → *Execute* with the pre-filled example → see `prediction_wh`, `model_used`, `routing_reason`.
3. Try `POST /predict?mode=rf` and `?mode=dt` (forced modes used for experiments).

📸 **SS-10** — Swagger `/docs` page. **SS-11** — a successful `/predict` response.

Open a **second** terminal (activate venv again) and run:
```powershell
pytest -v                                         # automated tests
python monitoring/smoke_test.py --url http://localhost:8080
python monitoring/load_test.py --url http://localhost:8080 --mode ecoroute --n 200 --workers 5 --label local
```
Then visit **http://localhost:8080/metrics** and **http://localhost:8080/drift**.

📸 **SS-12** — `pytest -v` all passed. **SS-13** — smoke test table (valid OK, invalid type/missing/out-of-range/extra field → 422). **SS-14** — `/metrics` JSON. *(Report: API, validation, security)*

---

## PHASE 5 — Drift experiment (Module 3: monitoring + drift)
```powershell
python -m monitoring.drift_experiment
```
Compares three datasets with the training baseline using PSI:
- **In-distribution** (training sample) → *Normal*
- **Real test period** → *Warning* — genuine **seasonal drift** (winter → spring). This is a real, explainable finding.
- **Simulated drift** (humidity and wind shifted) → very high PSI

📸 **SS-15** — the output table. *(Report: Drift monitoring)*

Say in the viva: *"Drift is a warning signal for investigation, not proof that the model failed."* (MAE for the drifted data is shown in the table so you can discuss this.)

---

## PHASE 6 — Docker (Module 2: containerisation)

Docker Desktop must be running.
```powershell
docker build -t ecoroute-ai:local .
docker run --rm -p 8080:8080 ecoroute-ai:local
```
(Stop the earlier `uvicorn` first, since both use port 8080.) Open http://localhost:8080/docs and run the smoke test again:
```powershell
python monitoring/smoke_test.py --url http://localhost:8080
docker images ecoroute-ai
```
📸 **SS-16** — `docker build` finishing successfully + `docker images` showing the image size. **SS-17** — `/docs` served from the container.

---

## PHASE 7 — GitHub (version control)

1. On github.com create an **empty** repository named `EcoRouteAI` (no README).
2. In the terminal:
```powershell
git init
git add .
git status
git commit -m "EcoRoute AI: models, router, API, Docker, CI/CD"
git branch -M main
git remote add origin https://github.com/<your-username>/EcoRouteAI.git
git push -u origin main
```
(Muskan: add as collaborator under *Settings → Collaborators*, then she can `git clone` it.)

📸 **SS-18** — GitHub repo page with the files. **SS-19** — `git status`/repo showing **no** `.env`, key files or passwords (security evidence).

---

## PHASE 8 — Google Cloud setup (Module 1: cloud foundations)

Replace `YOUR_PROJECT_ID` with a globally unique name, e.g. `ecoroute-ai-kowsya-01`.

```powershell
gcloud auth login
gcloud projects create YOUR_PROJECT_ID --name="EcoRoute AI"
gcloud config set project YOUR_PROJECT_ID
```
**Billing (console, required for Cloud Run):** https://console.cloud.google.com/billing → link your project to a billing account.
**Budget alert (console):** *Billing → Budgets & alerts → Create budget* → small amount, alerts at 50/90/100 %.
⚠️ A budget alert only **notifies** you; it does not stop spending. That is why we also set `--max-instances`. Check the current Cloud Run pricing/free tier on the official page before deploying.

📸 **SS-20** — budget alert page.

Enable the services and create the image repository:
```powershell
gcloud services enable run.googleapis.com artifactregistry.googleapis.com cloudbuild.googleapis.com logging.googleapis.com monitoring.googleapis.com
gcloud artifacts repositories create ecoroute-repo --repository-format=docker --location=asia-south1 --description="EcoRoute images"
gcloud auth configure-docker asia-south1-docker.pkg.dev
```
Create a **runtime service account with no extra permissions** (least privilege – the running container needs no access to other Google services):
```powershell
gcloud iam service-accounts create ecoroute-runtime --display-name="EcoRoute runtime (no roles)"
```
📸 **SS-21** — Artifact Registry page showing `ecoroute-repo`. **SS-22** — *IAM & Admin → Service Accounts* page showing `ecoroute-runtime`.

---

## PHASE 9 — Deploy to Cloud Run (Module 2: deployment)

```powershell
docker build -t asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/ecoroute-repo/ecoroute-ai:v1 .
docker push asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/ecoroute-repo/ecoroute-ai:v1
gcloud run deploy ecoroute-ai --image asia-south1-docker.pkg.dev/YOUR_PROJECT_ID/ecoroute-repo/ecoroute-ai:v1 --region asia-south1 --platform managed --allow-unauthenticated --max-instances 3 --concurrency 20 --cpu 1 --memory 512Mi --service-account ecoroute-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
The last command prints a **Service URL** (`https://ecoroute-ai-xxxx.a.run.app`). Test it:
```powershell
python monitoring/smoke_test.py --url https://ecoroute-ai-xxxx.a.run.app
```
Open `<URL>/docs` in the browser (HTTPS is automatic).

> `--allow-unauthenticated` makes the demo reachable by your teacher. If you prefer a private service, remove it and call with an identity token; mention this trade-off in the security section.

📸 **SS-23** — terminal showing deploy success + Service URL. **SS-24** — Cloud Run console → service page (revision, region, max instances, concurrency). **SS-25** — `<URL>/docs` in browser (shows `https://`). **SS-26** — smoke test against the Cloud Run URL. *(Report: Cloud deployment)*

**Day-8 milestone:** core project online.

---

## PHASE 10 — CI/CD with GitHub Actions (Module 2 practical 2)

Create a **deployer service account** with only the roles the pipeline needs:
```powershell
gcloud iam service-accounts create github-deployer --display-name="GitHub Actions deployer"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/run.admin"
gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/artifactregistry.writer"
gcloud iam service-accounts add-iam-policy-binding ecoroute-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com --member="serviceAccount:github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/iam.serviceAccountUser"
gcloud iam service-accounts keys create deployer-key.json --iam-account=github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
In GitHub: *Settings → Secrets and variables → Actions → New repository secret*:
- `GCP_PROJECT_ID` = your project id
- `GCP_SA_KEY` = the **entire contents** of `deployer-key.json`

Then **delete the key file locally** (never commit it): `del deployer-key.json`

Edit `.github/workflows/deploy.yml` only if your region/service names differ (the workflow deploys with `--service-account` default; if you want it to reuse `ecoroute-runtime`, add `--service-account ecoroute-runtime@YOUR_PROJECT_ID.iam.gserviceaccount.com` to the deploy step).

Trigger it by making a small change and pushing:
```powershell
git add .
git commit -m "Trigger CI/CD"
git push
```
GitHub → **Actions** tab → watch *test* then *deploy* turn green.

📸 **SS-27** — Actions run with both jobs green (click the run to show steps). **SS-28** — Cloud Run *Revisions* tab showing the new revision tagged with the commit id. *(Report: CI/CD pipeline)*

> Service-account keys are simple but less secure than Workload Identity Federation. Mention WIF as future work.

---

## PHASE 11 — Monitoring on GCP (Module 3 practical 1)

First generate traffic (small and controlled):
```powershell
python monitoring/load_test.py --url https://ecoroute-ai-xxxx.a.run.app --mode ecoroute --n 300 --workers 5 --label cloud-baseline
```

**11.1 Logs** — Console → *Cloud Run → ecoroute-ai → Logs*. You will see one JSON line per request (`event, request_id, model_used, latency_ms, status`). Notice **no input values** are logged (privacy).
📸 **SS-29** — logs view; expand one entry to show the fields.

**11.2 Log-based metrics** (turn logs into monitorable numbers):
```powershell
gcloud logging metrics create ecoroute_rf_requests --description="Requests routed to random forest" --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"ecoroute-ai\" AND jsonPayload.model_used=\"random_forest\""
gcloud logging metrics create ecoroute_dt_requests --description="Requests routed to decision tree" --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"ecoroute-ai\" AND jsonPayload.model_used=\"decision_tree\""
gcloud logging metrics create ecoroute_validation_errors --description="Rejected invalid inputs" --log-filter="resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"ecoroute-ai\" AND jsonPayload.event=\"validation_error\""
gcloud logging metrics create ecoroute_latency_ms --config-from-file=monitoring/latency_metric.yaml
```
> In PowerShell the `\"` escaping above works; if a command errors on quotes, create the same metric in the console: *Logging → Log-based Metrics → Create metric* using the filter text.
> New metrics only count logs produced **after** creation, so run the load test again afterwards.

**11.3 Cloud Run built-in metrics** — *Cloud Run → ecoroute-ai → Metrics tab*: request count, request latency (p50/p95/p99), container instance count, CPU utilisation, memory utilisation, billable instance time.

**11.4 Dashboard** — *Monitoring → Dashboards → Create dashboard*, add charts: Cloud Run request count, request latency p95, CPU utilisation, memory utilisation, `ecoroute_rf_requests` vs `ecoroute_dt_requests`, `ecoroute_latency_ms`.

**11.5 Application metrics and drift from the live service**
```powershell
curl.exe https://ecoroute-ai-xxxx.a.run.app/metrics
curl.exe https://ecoroute-ai-xxxx.a.run.app/drift
```
Then send **drifted** traffic and call `/drift` again (see below).

📸 **SS-30** — Cloud Run Metrics tab. **SS-31** — your custom dashboard. **SS-32** — `/metrics` from the cloud URL. **SS-33** — `/drift` response. *(Report: Model & application monitoring)*

**Live drift demo on the cloud:** send artificially shifted traffic with the `--drift` option:
```powershell
python monitoring/load_test.py --url https://ecoroute-ai-xxxx.a.run.app --mode ecoroute --n 100 --workers 5 --label drift-demo --drift
```
Then open `<URL>/drift`: status should change to **Warning**.
📸 **SS-34** — `/drift` showing Warning after drifted traffic.

---

## PHASE 12 — Scalability experiments (Module 3 practical 2)

Run **small, controlled** tests (keep `--n` ≤ 500). Change settings with `gcloud run services update`, then run the load test with a `--label` so each run is recorded in `results/load_test_log.csv`. Wait ~1 minute after each update.

**Experiment A — max instances (use low concurrency so scaling is needed)**
```powershell
gcloud run services update ecoroute-ai --region asia-south1 --concurrency 5 --max-instances 1
python monitoring/load_test.py --url <URL> --mode ecoroute --n 400 --workers 30 --label "A maxinst=1 conc=5"
gcloud run services update ecoroute-ai --region asia-south1 --concurrency 5 --max-instances 5
python monitoring/load_test.py --url <URL> --mode ecoroute --n 400 --workers 30 --label "A maxinst=5 conc=5"
```
**Experiment B — concurrency (max instances fixed at 3)**
```powershell
gcloud run services update ecoroute-ai --region asia-south1 --max-instances 3 --concurrency 2
python monitoring/load_test.py --url <URL> --mode ecoroute --n 400 --workers 30 --label "B conc=2"
gcloud run services update ecoroute-ai --region asia-south1 --max-instances 3 --concurrency 40
python monitoring/load_test.py --url <URL> --mode ecoroute --n 400 --workers 30 --label "B conc=40"
```
**Experiment C — cold start** (service must scale to zero: `--min-instances 0` is the default)
```powershell
gcloud run services update ecoroute-ai --region asia-south1 --min-instances 0 --concurrency 20 --max-instances 3
# wait at least 15 minutes without sending any request, then:
python monitoring/load_test.py --url <URL> --mode ecoroute --n 1 --workers 1 --label "C cold first request"
python monitoring/load_test.py --url <URL> --mode ecoroute --n 20 --workers 1 --label "C warm requests"
```
**Experiment D — DT vs RF vs EcoRoute in the cloud** (same settings for all three)
```powershell
python monitoring/load_test.py --url <URL> --mode dt --n 400 --workers 10 --label "D dt"
python monitoring/load_test.py --url <URL> --mode rf --n 400 --workers 10 --label "D rf"
python monitoring/load_test.py --url <URL> --mode ecoroute --n 400 --workers 10 --label "D ecoroute"
```
📸 **SS-35** — terminal output of experiment A (both runs). **SS-36** — B. **SS-37** — C (cold vs warm). **SS-38** — D. **SS-39** — Cloud Run Metrics tab (instance count rising during experiment A). **SS-40** — `results/load_test_log.csv` opened in VS Code. *(Report: Scalability)*

> Client-side latency includes your internet connection, so differences between runs can be partly network noise. Mention this limitation and compare runs done close together. Don't claim exact production numbers.

---

## PHASE 13 — Cost analysis (Module 3 practical 2)

1. From `results/final_comparison.csv` use the **estimated compute cost per 1000 requests** (local model-execution time × price). **Update the three price constants at the top of `src/evaluate.py` from the official Cloud Run pricing page for your region, then re-run `python -m src.evaluate`.**
2. In the console open *Cloud Run → Metrics* (Billable instance time, CPU, memory) and *Billing → Reports* (filter: Cloud Run). With a few hundred requests the real bill will probably be near zero or inside the free tier; that is fine. Report the **calculated estimate with assumptions** and the real billing screenshot.
3. Make the cost table: DT only / RF only / EcoRoute → RF share %, avg latency, estimated cost per 1000 requests, billable instance time observed in experiment D.

📸 **SS-41** — re-run `evaluate` with updated prices. **SS-42** — Billing report / budget page. *(Report: Cost optimisation)*

**Correct claim:** *"Routing reduced the proportion of requests sent to the more expensive model; under our configuration this changed estimated compute time by X %."* **Never** say "we saved 76 % cost" just because 76 % of requests went to the DT.

---

## PHASE 14 — Security and privacy evidence (Module 3)

```powershell
gcloud run services get-iam-policy ecoroute-ai --region asia-south1
gcloud run services describe ecoroute-ai --region asia-south1
```
- Input validation → **SS-13/SS-26** (smoke tests).
- Secrets not in GitHub → **SS-19** (and GitHub search for `key` returns nothing).
- Logs contain no payloads → **SS-29**.
- HTTPS → **SS-25**.
- IAM least privilege (runtime SA has no roles; deployer SA has 3 specific roles) → 📸 **SS-43** *IAM & Admin → IAM* page.
- Resource limits (`max-instances`) → **SS-24**.
- GitHub Secrets configured (names only, values hidden) → 📸 **SS-44**.

Privacy statement for the report: *"The project uses a public research dataset and does not collect personally identifiable information. The API accepts only the nine fields needed for prediction, rejects extra fields, and logs request ID, model used, latency and status — not input values. The design is GDPR-inspired; we do not claim GDPR certification. India's DPDP Act, 2023 is relevant for real household data, subject to the provisions in force."*

---

## PHASE 15 — Clean up (avoid charges after the demo)

```powershell
gcloud run services delete ecoroute-ai --region asia-south1
gcloud artifacts repositories delete ecoroute-repo --location=asia-south1
gcloud iam service-accounts keys list --iam-account=github-deployer@YOUR_PROJECT_ID.iam.gserviceaccount.com
```
Delete the deployer key in the console if no longer needed. Only do this **after** your report screenshots and demo are finished.

---

## Syllabus mapping (for the report)

| Module | Practical / topic | Where in project | Evidence |
|---|---|---|---|
| 1 | ML environment on cloud, AI lifecycle | Phases 2–3, 8 | SS-03…SS-09, SS-20…SS-22 |
| 2 | Containerise + REST API | Phases 4, 6, 9 | SS-10…SS-17, SS-23…SS-26 |
| 2 | CI/CD pipeline | Phase 10 | SS-27, SS-28 |
| 3 | Monitoring, drift, latency | Phases 5, 11 | SS-15, SS-29…SS-34 |
| 3 | Cost and scalability analysis + optimisation | Phases 12–13 | SS-35…SS-42 |
| 3 | Ethical AI, security, privacy | Phase 14 | SS-13, SS-19, SS-43, SS-44 |
| 4 (conceptual) | AI sustainability | Routing avoids unnecessary expensive computation | Discussion in report |

## Viva cheat-sheet — "explain this code"

- **`config.py`** – one source of truth for features/paths so training and API can never disagree (prevents training/serving skew).
- **`preprocessing.py`** – loads CSV, extracts `hour`/`dayofweek`, splits **chronologically** so no future data leaks into training.
- **`feature_selection_cv.py`** – forward-in-time CV on train+val showed absolute-temperature features hurt on future data (seasonal shift), so we excluded them without using the test set.
- **`train_models.py`** – small grid search scored on validation MAE; DT = cheap, RF = 100 trees = costlier.
- **`train_router.py`** – label = "was RF better than DT by more than a margin?"; tiny depth-3 classifier trained on validation predictions; margin chosen by `routed_MAE + cost_weight × RF_share` using forward-in-time CV.
- **`evaluate.py`** – replays every test row one at a time through DT-only, RF-only, EcoRoute and measures MAE, RMSE, avg/p95 latency, throughput, RF share, cost estimate.
- **`drift.py`** – PSI: compare the share of requests in each training-quantile bin; <0.1 normal, 0.1–0.25 moderate, ≥0.25 warning; a signal, not proof of failure.
- **`api/main.py`** – Pydantic validation (422 on bad input), router → model → response, JSON logs without payload, `/metrics`, `/drift`, `/health`, generic error handler hides tracebacks.
- **`Dockerfile`** – slim Python, dependencies in a cached layer, non-root user, listens on `$PORT`.
- **`deploy.yml`** – on push: run pytest → build image → push to Artifact Registry → deploy to Cloud Run with `max-instances`/`concurrency`.

## Honest limitations (put in the report)
Single building and one 4.5-month period; absolute-temperature features excluded due to seasonal shift; router trained on ~3000 validation rows; accuracy differences between approaches are small and come from one test period; client-side latency in cloud tests includes network time; cost figures are estimates from assumed prices; in-memory metrics reset when an instance restarts; predictions are informational estimates, not guaranteed savings.
