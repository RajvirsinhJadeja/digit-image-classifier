import sqlite3
import pickle

class SqliteDict:
    def __init__(self, db_file="data/word_map.db") -> None:
        self.conn = sqlite3.connect(db_file)
        self.cursor = self.conn.cursor()
    
    def __getitem__(self, key):
        self.cursor.execute("SELECT value FROM kv WHERE key = ?", (key,))
        row = self.cursor.fetchone()
        
        if row:
            return pickle.loads(row[0])
        else:
            return None
    
    def close(self):
        self.conn.close()