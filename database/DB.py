import sqlite3

class Database:
    def __init__(self,sqlite_file):
        self._conn = sqlite3.connect(sqlite_file)
        self._cursor = self._conn.cursor()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.commit()
        self.connection.close()
    
    @property
    def connection(self):
        return self._conn
    
    @property
    def cursor(self):
        return self._cursor
    
    def commit(self):
        self.connection.commit()
    
    def execmany(self, sql, params):
        print(sql)
        print(str(params))
        self.cursor.executemany(sql, params or ())

    def execute(self, sql, params=None):
        self.cursor.execute(sql, params or ())
    
    def fetchall(self):
        return self.cursor.fetchall()
    
    def fetchone(self):
        return self.cursor.fetchone()
    
    def query(self, sql, params=None):
        self.cursor.execute(sql, params or ())
        return self.fetchall()
