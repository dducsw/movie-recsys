import os
import argparse
import pandas as pd
import psycopg2

def import_movies(csv_path, host, port, database, user, password):
    """
    Load movie CSV dataset and import it into PostgreSQL using raw psycopg2.
    """
    if not os.path.exists(csv_path):
        print(f"[Error] Movie CSV file not found at: {csv_path}")
        return
        
    print(f"Reading movie dataset from: {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"[Error] Failed to read CSV: {e}")
        return
        
    num_movies = len(df)
    print(f"Loaded {num_movies} movies from CSV. Connecting to PostgreSQL...")
    
    try:
        # Connect to Postgres database directly
        conn = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password
        )
        cur = conn.cursor()
        
        # Drop and create movies table
        print("Re-creating 'movies' table...")
        cur.execute("DROP TABLE IF EXISTS movies;")
        cur.execute("""
            CREATE TABLE movies (
                movieId INT PRIMARY KEY,
                title VARCHAR(500),
                release_date VARCHAR(50),
                genres VARCHAR(500),
                popularity DOUBLE PRECISION,
                adult BOOLEAN,
                overview TEXT,
                vote_average DOUBLE PRECISION,
                vote_count INT,
                poster_url VARCHAR(500)
            );
        """)
        
        # Clean and prepare records for insert
        df["adult"] = df["adult"].astype(bool)
        df["movieId"] = df["movieId"].astype(int)
        df["vote_count"] = df["vote_count"].fillna(0).astype(int)
        df["vote_average"] = df["vote_average"].fillna(0.0).astype(float)
        df["popularity"] = df["popularity"].fillna(0.0).astype(float)
        df["title"] = df["title"].fillna("")
        df["release_date"] = df["release_date"].fillna("")
        df["genres"] = df["genres"].fillna("")
        df["overview"] = df["overview"].fillna("")
        df["poster_url"] = df["poster_url"].fillna("")
        
        records = df[[
            "movieId", "title", "release_date", "genres", "popularity", 
            "adult", "overview", "vote_average", "vote_count", "poster_url"
        ]].values.tolist()
        
        # Bulk Insert using executemany
        print("Uploading data to PostgreSQL...")
        insert_query = """
            INSERT INTO movies (
                movieId, title, release_date, genres, popularity, 
                adult, overview, vote_average, vote_count, poster_url
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cur.executemany(insert_query, records)
        conn.commit()
        
        cur.close()
        conn.close()
        print(f"\n[Success] Successfully imported {num_movies} movies into table 'movies' in database '{database}'!")
        
    except Exception as e:
        print(f"[Error] Database operation failed: {e}")
        print("\nSuggestions:")
        print("1. Ensure your Postgres Docker container is running: 'docker-compose up -d'")
        print("2. Check if the database port (5435) matches your docker-compose.yml configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Crawled Movies CSV to PostgreSQL using Psycopg2")
    parser.add_argument("--csv_path", type=str, default="data/crawler/movies_crawled.csv",
                        help="Path to crawled movies CSV (default: data/crawler/movies_crawled.csv)")
    parser.add_argument("--host", type=str, default="localhost",
                        help="PostgreSQL host (default: localhost)")
    parser.add_argument("--port", type=int, default=5435,
                        help="PostgreSQL port (default: 5435)")
    parser.add_argument("--database", type=str, default="movie_db",
                        help="PostgreSQL database name (default: movie_db)")
    parser.add_argument("--user", type=str, default="postgres",
                        help="PostgreSQL user (default: postgres)")
    parser.add_argument("--password", type=str, default="mysecretpassword",
                        help="PostgreSQL password (default: mysecretpassword)")
                        
    args = parser.parse_args()
    
    import_movies(
        csv_path=args.csv_path,
        host=args.host,
        port=args.port,
        database=args.database,
        user=args.user,
        password=args.password
    )
