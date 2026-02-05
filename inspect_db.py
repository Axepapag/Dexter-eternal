import sqlite3
import json
from pathlib import Path

db_path = Path(r'D:\Dexter-Eternal\brain.db')
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [row[0] for row in cursor.fetchall()]

print('=== Brain Database Tables ===')
for table in tables:
    print(f'\n{table}:')
    cursor.execute(f'SELECT COUNT(*) FROM {table}')
    count = cursor.fetchone()[0]
    print(f'  Total records: {count}')
    
    cursor.execute(f'PRAGMA table_info({table})')
    columns = cursor.fetchall()
    print(f'  Columns: {", ".join([col[1] for col in columns])}')

conn.close()
