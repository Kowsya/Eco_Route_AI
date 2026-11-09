# EcoRoute AI
Cost-aware intelligent model routing (Decision Tree vs Random Forest) for cloud-based smart-home
energy prediction. CSI-6402 SEA project.

Start with **GUIDE.md** – it has every command in order and the screenshot checklist.

Quick start:
    python -m venv venv && venv\Scripts\activate
    pip install -r requirements.txt
    python -m src.train_models && python -m src.train_router && python -m src.evaluate
    uvicorn api.main:app --port 8080
