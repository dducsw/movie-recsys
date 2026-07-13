import os
import argparse
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

def import_movies(csv_path, host, port, database, user, password):
    """
    Load movie CSV dataset and import it into PostgreSQL using raw psycopg2 with a normalized schema.
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
        
        # 1. Drop existing tables (with cascade to clean up dependencies)
        print("Dropping existing tables if they exist...")
        cur.execute("DROP TABLE IF EXISTS movie_genres CASCADE;")
        cur.execute("DROP TABLE IF EXISTS movie_actors CASCADE;")
        cur.execute("DROP TABLE IF EXISTS movie_tags CASCADE;")
        cur.execute("DROP TABLE IF EXISTS movies CASCADE;")
        cur.execute("DROP TABLE IF EXISTS directors CASCADE;")
        cur.execute("DROP TABLE IF EXISTS genres CASCADE;")
        cur.execute("DROP TABLE IF EXISTS actors CASCADE;")
        cur.execute("DROP TABLE IF EXISTS tags CASCADE;")
        
        # 2. Create normalized tables
        print("Creating normalized schema tables...")
        cur.execute("""
            CREATE TABLE directors (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE genres (
                id SERIAL PRIMARY KEY,
                name VARCHAR(100) UNIQUE NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE actors (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE tags (
                id SERIAL PRIMARY KEY,
                name VARCHAR(255) UNIQUE NOT NULL
            );
        """)
        cur.execute("""
            CREATE TABLE movies (
                movieId INT PRIMARY KEY,
                title VARCHAR(500) NOT NULL,
                release_date VARCHAR(50),
                popularity DOUBLE PRECISION,
                adult BOOLEAN,
                overview TEXT,
                vote_average DOUBLE PRECISION,
                vote_count INT,
                poster_url VARCHAR(500),
                director_id INT REFERENCES directors(id) ON DELETE SET NULL,
                trailer_url VARCHAR(500)
            );
        """)
        cur.execute("""
            CREATE TABLE movie_genres (
                movie_id INT REFERENCES movies(movieId) ON DELETE CASCADE,
                genre_id INT REFERENCES genres(id) ON DELETE CASCADE,
                PRIMARY KEY (movie_id, genre_id)
            );
        """)
        cur.execute("""
            CREATE TABLE movie_actors (
                movie_id INT REFERENCES movies(movieId) ON DELETE CASCADE,
                actor_id INT REFERENCES actors(id) ON DELETE CASCADE,
                cast_order INT,
                PRIMARY KEY (movie_id, actor_id)
            );
        """)
        cur.execute("""
            CREATE TABLE movie_tags (
                movie_id INT REFERENCES movies(movieId) ON DELETE CASCADE,
                tag_id INT REFERENCES tags(id) ON DELETE CASCADE,
                PRIMARY KEY (movie_id, tag_id)
            );
        """)
        
        # Commit table creation before loading data
        conn.commit()
        print("Schema tables created successfully!")

        # 3. Extract unique master entities from CSV
        print("Extracting unique master entities (directors, genres, actors, tags)...")
        
        # Directors
        directors = sorted(list(df["director"].dropna().astype(str).unique()))
        
        # Genres
        genres_set = set()
        for genres_str in df["genres"].dropna().astype(str):
            for g in genres_str.split("|"):
                g_stripped = g.strip()
                if g_stripped:
                    genres_set.add(g_stripped)
        genres = sorted(list(genres_set))
        
        # Actors
        actors_set = set()
        for cast_str in df["cast"].dropna().astype(str):
            for a in cast_str.split("|"):
                a_stripped = a.strip()
                if a_stripped:
                    actors_set.add(a_stripped)
        actors = sorted(list(actors_set))
        
        # Tags (Keywords)
        tags_set = set()
        for kw_str in df["keywords"].dropna().astype(str):
            for t in kw_str.split("|"):
                t_stripped = t.strip()
                if t_stripped:
                    tags_set.add(t_stripped)
        tags = sorted(list(tags_set))

        # 4. Insert master entities in bulk
        print(f"Uploading {len(directors)} directors...")
        execute_values(cur, "INSERT INTO directors (name) VALUES %s ON CONFLICT (name) DO NOTHING", [(d,) for d in directors])
        
        print(f"Uploading {len(genres)} genres...")
        execute_values(cur, "INSERT INTO genres (name) VALUES %s ON CONFLICT (name) DO NOTHING", [(g,) for g in genres])
        
        print(f"Uploading {len(actors)} actors...")
        execute_values(cur, "INSERT INTO actors (name) VALUES %s ON CONFLICT (name) DO NOTHING", [(a,) for a in actors])
        
        print(f"Uploading {len(tags)} tags...")
        execute_values(cur, "INSERT INTO tags (name) VALUES %s ON CONFLICT (name) DO NOTHING", [(t,) for t in tags])
        
        conn.commit()

        # 5. Retrieve IDs to build mappings
        print("Building ID maps in memory...")
        cur.execute("SELECT id, name FROM directors")
        director_to_id = {name: id for id, name in cur.fetchall()}
        
        cur.execute("SELECT id, name FROM genres")
        genre_to_id = {name: id for id, name in cur.fetchall()}
        
        cur.execute("SELECT id, name FROM actors")
        actor_to_id = {name: id for id, name in cur.fetchall()}
        
        cur.execute("SELECT id, name FROM tags")
        tag_to_id = {name: id for id, name in cur.fetchall()}

        # 6. Clean and prepare movies records
        print("Preparing movies data...")
        df["adult"] = df["adult"].fillna(False).astype(bool)
        df["movieId"] = df["movieId"].astype(int)
        df["vote_count"] = df["vote_count"].fillna(0).astype(int)
        df["vote_average"] = df["vote_average"].fillna(0.0).astype(float)
        df["popularity"] = df["popularity"].fillna(0.0).astype(float)
        df["title"] = df["title"].fillna("").astype(str)
        df["release_date"] = df["release_date"].fillna("").astype(str)
        df["overview"] = df["overview"].fillna("").astype(str)
        df["poster_url"] = df["poster_url"].fillna("").astype(str)

        movie_records = []
        for _, row in df.iterrows():
            d_name = str(row["director"]).strip() if pd.notna(row["director"]) else None
            d_id = director_to_id.get(d_name) if d_name and d_name in director_to_id else None
            
            movie_records.append((
                int(row["movieId"]),
                str(row["title"]),
                str(row["release_date"]),
                float(row["popularity"]),
                bool(row["adult"]),
                str(row["overview"]),
                float(row["vote_average"]),
                int(row["vote_count"]),
                str(row["poster_url"]),
                d_id,
                None # trailer_url is default to None and lazy loaded
            ))

        print(f"Uploading {len(movie_records)} movies to 'movies' table...")
        execute_values(cur, """
            INSERT INTO movies (
                movieId, title, release_date, popularity, adult, 
                overview, vote_average, vote_count, poster_url, director_id, trailer_url
            ) VALUES %s ON CONFLICT (movieId) DO NOTHING
        """, movie_records)
        
        conn.commit()

        # 7. Prepare and insert junction records
        print("Preparing junction relationships...")
        
        movie_genres_records = []
        movie_actors_records = []
        movie_tags_records = []
        
        for _, row in df.iterrows():
            m_id = int(row["movieId"])
            
            # Genres
            genres_str = row.get("genres")
            if pd.notna(genres_str):
                for g in str(genres_str).split("|"):
                    g_stripped = g.strip()
                    if g_stripped and g_stripped in genre_to_id:
                        movie_genres_records.append((m_id, genre_to_id[g_stripped]))
            
            # Cast
            cast_str = row.get("cast")
            if pd.notna(cast_str):
                seen_actors_for_movie = set()
                for order, actor in enumerate(str(cast_str).split("|")):
                    actor_stripped = actor.strip()
                    if actor_stripped and actor_stripped in actor_to_id:
                        a_id = actor_to_id[actor_stripped]
                        if a_id not in seen_actors_for_movie:
                            seen_actors_for_movie.add(a_id)
                            movie_actors_records.append((m_id, a_id, order))
            
            # Keywords
            kw_str = row.get("keywords")
            if pd.notna(kw_str):
                for tag in str(kw_str).split("|"):
                    tag_stripped = tag.strip()
                    if tag_stripped and tag_stripped in tag_to_id:
                        movie_tags_records.append((m_id, tag_to_id[tag_stripped]))

        # Deduplicate sets
        movie_genres_records = list(set(movie_genres_records))
        movie_tags_records = list(set(movie_tags_records))
        
        print(f"Uploading {len(movie_genres_records)} movie-genre relationships...")
        execute_values(cur, "INSERT INTO movie_genres (movie_id, genre_id) VALUES %s ON CONFLICT DO NOTHING", movie_genres_records)
        
        print(f"Uploading {len(movie_actors_records)} movie-actor relationships...")
        execute_values(cur, "INSERT INTO movie_actors (movie_id, actor_id, cast_order) VALUES %s ON CONFLICT DO NOTHING", movie_actors_records)
        
        print(f"Uploading {len(movie_tags_records)} movie-tag relationships...")
        execute_values(cur, "INSERT INTO movie_tags (movie_id, tag_id) VALUES %s ON CONFLICT DO NOTHING", movie_tags_records)
        
        conn.commit()
        
        cur.close()
        conn.close()
        print(f"\n[Success] Rebuilt database and successfully imported {num_movies} movies with normalized genres, actors, tags, and directors!")
        
    except Exception as e:
        print(f"[Error] Database operation failed: {e}")
        print("\nSuggestions:")
        print("1. Ensure your Postgres Docker container is running: 'docker-compose up -d'")
        print("2. Check if the database port (5435) matches your docker-compose.yml configuration.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import Crawled Movies CSV to PostgreSQL using Psycopg2 with normalized schema")
    parser.add_argument("--csv_path", type=str, default="data/crawler/movies_crawled_10k.csv",
                        help="Path to crawled movies CSV (default: data/crawler/movies_crawled_10k.csv)")
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
