# 🐳 Hạ tầng Docker & MLOps (Docker Infrastructure)

Thư mục `docker/` chứa các tệp cấu hình, kịch bản khởi tạo và Dockerfile cho toàn bộ các dịch vụ hạ tầng phụ trợ, cơ sở dữ liệu, bộ nhớ đệm, kho vector, nền tảng theo dõi thí nghiệm và hệ thống giám sát của **MovieNex**.

---

## 🗂️ Cấu trúc Thư mục

```
docker/
├── README.md                 # Tài liệu này
├── mlflow/
│   └── Dockerfile            # Dockerfile cài đặt MLflow server kèm driver psycopg2 & boto3
├── postgres/
│   └── init.sql              # Script SQL khởi tạo các cơ sở dữ liệu bổ sung (ví dụ: mlflow_db)
├── prometheus/
│   └── prometheus.yml        # Cấu hình chu kỳ scrape số liệu từ FastAPI backend (/metrics)
├── qdrant/
│   └── config.yaml           # Cấu hình Vector Database Qdrant
├── redis/
│   └── redis.conf            # Cấu hình tối ưu hóa bộ nhớ đệm và lưu trữ Redis AOF/RDB
└── seaweedfs/
    └── create-buckets.sh     # Script tự động tạo các S3 buckets (mlflow, recsys-data, feature-store)
```

---

## 🏗️ Bảng Tổng hợp Dịch vụ trong Docker Compose

Tất cả các dịch vụ được điều phối thống nhất thông qua file [`docker-compose.yml`](../docker-compose.yml) tại thư mục gốc:

| Tên Service | Container Name | Image / Build | Port ánh xạ | Chức năng |
| :--- | :--- | :--- | :--- | :--- |
| **postgres** | `movie-recsys-db` | `postgres:15-alpine` | `5435:5432` | CSDL quan hệ chính (`movie_db`) & backend store cho MLflow (`mlflow_db`) |
| **redis** | `movie-recsys-redis` | `redis:7-alpine` | `6379:6379` | Cache tốc độ cao, online feature store & Redis Stream |
| **qdrant** | `movie-recsys-qdrant` | `qdrant/qdrant:latest` | `6333:6333`, `6334:6334` | Vector Database cho tìm kiếm tương đồng đa chiều |
| **seaweedfs** | `movie-recsys-seaweedfs` | `chrislusf/seaweedfs:latest` | `8333:8333` (S3), `8888:8888` (Filer) | Lưu trữ Object tương thích AWS S3 cho model artifacts |
| **seaweedfs-init** | `movie-recsys-seaweedfs-init` | `amazon/aws-cli:latest` | N/A (One-shot) | Tự động tạo các bucket: `mlflow`, `recsys-data`, `feature-store` |
| **mlflow** | `movie-recsys-mlflow` | `./docker/mlflow/Dockerfile` | `5000:5000` | Giao diện theo dõi thí nghiệm & đăng ký mô hình |
| **prometheus** | `movie-recsys-prometheus` | `prom/prometheus:latest` | `9090:9090` | Thu thập và lưu trữ số liệu hiệu năng hệ thống |
| **backend** *(profile: app)* | `movie-recsys-backend` | `./web_app/backend/Dockerfile` | `8000:8000` | FastAPI WebApp Gateway & Recommendation Server |
| **frontend** *(profile: app)* | `movie-recsys-frontend` | `./web_app/frontend/Dockerfile` | `5173:80` | Giao diện người dùng React phục vụ qua Nginx |

---

## 🚀 Hướng dẫn Quản lý Hạ tầng Chi tiết

Chạy các lệnh sau từ **thư mục gốc dự án** (`movie-recsys/`):

### 1. Khởi động Dịch vụ Cốt lõi (Khuyên dùng khi Dev cục bộ)
Chỉ khởi động các dịch vụ lưu trữ dữ liệu và cache, sau đó chạy backend và frontend trực tiếp trên máy để dễ debug:
```bash
docker compose up -d postgres redis qdrant seaweedfs
```

### 2. Khởi động Toàn bộ MLOps Stack
Bao gồm cả MLflow, Prometheus và SeaweedFS:
```bash
docker compose up -d postgres redis qdrant seaweedfs seaweedfs-init mlflow prometheus
```

### 3. Khởi động Toàn bộ Hệ thống bao gồm WebApp (Full Containerized Stack)
Sử dụng cờ `--profile app` để bật cả backend và frontend trong Docker:
```bash
docker compose --profile app up -d --build
```

### 4. Kiểm tra Trạng thái & Sức khỏe Containers
```bash
docker compose ps
```

### 5. Xem Nhật ký Hoạt động (Logs)
```bash
# Xem log toàn bộ hệ thống
docker compose logs -f

# Xem log riêng một dịch vụ
docker compose logs -f postgres
docker compose logs -f mlflow
docker compose logs -f qdrant
```

### 6. Dừng Hạ tầng
```bash
# Dừng các container nhưng giữ nguyên dữ liệu
docker compose stop

# Dừng và xóa container (Dữ liệu vẫn được lưu trong Docker Volumes)
docker compose down

# Dừng và XÓA TOÀN BỘ DỮ LIỆU VOLUMES (Cẩn trọng!)
docker compose down -v
```

---

## 💾 Quản lý Dữ liệu Bền vững (Persistent Volumes)

Dữ liệu của các container được gắn vào các Docker Named Volumes:
* `pgdata`: Toàn bộ dữ liệu bảng của PostgreSQL.
* `redisdata`: Snapshots và log AOF của Redis.
* `qdrantdata`: Bộ sưu tập vector embeddings của Qdrant.
* `seaweeddata`: Toàn bộ tệp và model artifacts trong SeaweedFS S3.
* `prometheusdata`: Cơ sở dữ liệu chuỗi thời gian (TSDB) của Prometheus.
