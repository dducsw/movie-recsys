#!/usr/bin/env bash
# start-webapp.sh
# Khởi động toàn bộ WebApp (Docker infra + FastAPI backend + React Vite frontend)

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_EXEC="/home/mercury/miniconda3/envs/ml-env/bin/python"

if [ ! -f "$PYTHON_EXEC" ]; then
    PYTHON_EXEC="python3"
fi

echo -e "\033[1;36m========================================================\033[0m"
echo -e "\033[1;36m   MOVIE RECOMMENDATION SYSTEM - WEBAPP STARTUP         \033[0m"
echo -e "\033[1;36m========================================================\033[0m"

# 1. Start Docker Infrastructure
echo -e "\n\033[1;33m[1/3] Khởi động Docker Containers (PostgreSQL, Redis, Qdrant)...\033[0m"
docker compose -f "$PROJECT_ROOT/docker-compose.yml" up -d postgres redis qdrant

echo -e "\033[1;32m[✓] Docker infra đã sẵn sàng.\033[0m"

# 2. Khởi chạy Backend
echo -e "\n\033[1;33m[2/3] Khởi chạy Backend FastAPI (port 8000)...\033[0m"
cd "$PROJECT_ROOT/apps/api"
"$PYTHON_EXEC" -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo -e "\033[1;32m[✓] Backend đã chạy (PID: $BACKEND_PID) -> http://localhost:8000\033[0m"
echo -e "\033[1;32m    Swagger API Docs: http://localhost:8000/docs\033[0m"

# 3. Khởi chạy Frontend
echo -e "\n\033[1;33m[3/3] Khởi chạy Frontend Vite (port 5173)...\033[0m"
cd "$PROJECT_ROOT/apps/web"
if [ ! -d "node_modules" ]; then
    echo "Đang cài đặt dependencies npm..."
    npm install
fi
npm run dev &
FRONTEND_PID=$!
echo -e "\033[1;32m[✓] Frontend đã chạy (PID: $FRONTEND_PID) -> http://localhost:5173\033[0m"

echo -e "\n\033[1;32m========================================================\033[0m"
echo -e "\033[1;32m  TẤT CẢ DỊCH VỤ ĐÃ HOẠT ĐỘNG!                         \033[0m"
echo -e "\033[1;32m========================================================\033[0m"
echo -e "  - Frontend UI    : \033[1;34mhttp://localhost:5173\033[0m"
echo -e "  - Backend API   : \033[1;34mhttp://localhost:8000\033[0m"
echo -e "  - API Swagger UI : \033[1;34mhttp://localhost:8000/docs\033[0m"
echo -e "  - Qdrant UI      : \033[1;34mhttp://localhost:6333/dashboard\033[0m"
echo -e "\nNhấn \033[1;31mCtrl + C\033[0m để dừng toàn bộ ứng dụng.\n"

# Dọn dẹp tiến trình khi thoát (Ctrl + C)
cleanup() {
    echo -e "\n\033[1;33mĐang dừng Backend (PID: $BACKEND_PID) và Frontend (PID: $FRONTEND_PID)...\033[0m"
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo -e "\033[1;32mĐã tắt thành công.\033[0m"
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
