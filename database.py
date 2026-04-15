import sqlite3
import os
from dotenv import load_dotenv

load_dotenv()

# sqlite db path - stored in project folder locally, /home/user/app on hf spaces
DB_PATH = os.getenv('DB_PATH', os.path.join(os.path.dirname(__file__), 'anomalyx.db'))


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename VARCHAR(255),
                columns_analyzed VARCHAR(500),
                total_rows INTEGER,
                anomalies_found INTEGER,
                methods_used VARCHAR(255),
                contamination REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("database ready!")
    except Exception as e:
        print(f"db error: {e}")


def save_scan(filename, columns, total_rows, anomalies_found, methods_used, contamination):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO scans (filename, columns_analyzed, total_rows, anomalies_found, methods_used, contamination) VALUES (?, ?, ?, ?, ?, ?)",
            (filename, columns, total_rows, anomalies_found, methods_used, contamination)
        )
        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"error saving scan: {e}")


def get_history():
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT 20")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        results = []
        for row in rows:
            row_dict = dict(row)
            if row_dict.get("created_at"):
                row_dict["created_at"] = str(row_dict["created_at"])
            results.append(row_dict)
        return results
    except Exception as e:
        print(f"error getting history: {e}")
        return []
