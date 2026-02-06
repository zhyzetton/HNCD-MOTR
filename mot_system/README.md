- 启动前端
```bash
cd frontend
pnpm dev
```
- 启动后端
```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```