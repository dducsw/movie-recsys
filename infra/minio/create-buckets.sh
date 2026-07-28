#!/bin/sh
# Create default S3 buckets in MinIO for Development Environment
/usr/bin/mc alias set myminio http://minio:9000 minioadmin minioadminpassword;
/usr/bin/mc mb myminio/mlflow --ignore-existing;
/usr/bin/mc mb myminio/recsys-data --ignore-existing;
/usr/bin/mc mb myminio/feature-store --ignore-existing;
echo "MinIO buckets created successfully!";
exit 0;
