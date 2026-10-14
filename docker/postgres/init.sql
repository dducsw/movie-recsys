-- ==============================================================================
-- Initialization script for PostgreSQL database & Least-Privilege Application Role
-- ==============================================================================

-- 1. Create Application Databases if they don't already exist
SELECT 'CREATE DATABASE movie_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'movie_db')\gexec

SELECT 'CREATE DATABASE mlflow_db'
WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'mlflow_db')\gexec

-- 2. Create Dedicated Least-Privilege Application User
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'movienex_app') THEN
        CREATE ROLE movienex_app WITH LOGIN PASSWORD 'movienex_app_secure_password';
    END IF;
END
$$;

-- 3. Grant Connection Privileges
GRANT CONNECT ON DATABASE movie_db TO movienex_app;

-- 4. Connect to movie_db and configure table/sequence privileges
\c movie_db

GRANT USAGE ON SCHEMA public TO movienex_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO movienex_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO movienex_app;

-- 5. Automatically grant DML privileges on future tables created by migrations
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO movienex_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT USAGE, SELECT ON SEQUENCES TO movienex_app;
