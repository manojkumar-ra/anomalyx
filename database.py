import mysql.connector
import os

# mysql for saving scan history

def get_connection():
    return mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", ""),
        database=os.getenv("MYSQL_DATABASE", "anomalyx")
    )


def init_db():
    # need to connect without db first to create it
    conn = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST", "localhost"),
        user=os.getenv("MYSQL_USER", "root"),
        password=os.getenv("MYSQL_PASSWORD", "")
    )
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE IF NOT EXISTS anomalyx")
    conn.close()

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INT AUTO_INCREMENT PRIMARY KEY,
            filename VARCHAR(255),
            columns_analyzed VARCHAR(500),
            total_rows INT,
            anomalies_found INT,
            methods_used VARCHAR(255),
            contamination FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()


def save_scan(filename, columns, total_rows, anomalies_found, methods_used, contamination):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO scans (filename, columns_analyzed, total_rows, anomalies_found, methods_used, contamination) VALUES (%s, %s, %s, %s, %s, %s)",
        (filename, columns, total_rows, anomalies_found, methods_used, contamination)
    )
    conn.commit()
    conn.close()


def get_history():
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM scans ORDER BY created_at DESC LIMIT 20")
    rows = cursor.fetchall()
    conn.close()

    # convert datetime to string for json
    for row in rows:
        if row.get("created_at"):
            row["created_at"] = row["created_at"].strftime("%Y-%m-%d %H:%M:%S")
    return rows
