import os
from typing import Dict, Optional
from llama_index.graph_stores.memgraph import MemgraphPropertyGraphStore
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()  # optional if you keep creds in .env

GRAPH_URL = os.getenv("MEMGRAPH_URL", "bolt://localhost:7687")
GRAPH_USER = os.getenv("MEMGRAPH_USER", "memgraph")
GRAPH_PASSWORD = os.getenv("MEMGRAPH_PASSWORD", "")

MYSQL_CONFIG = {
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "root"),
    "host": os.getenv("MYSQL_HOST", "host.docker.internal"),
    "database": os.getenv("MYSQL_DB", "employees"),
}

QUERY_RESET = """
MATCH (n)
DETACH DELETE n
"""

QUERY_1 = """
CALL migrate.mysql(
  'SELECT DISTINCT TABLE_NAME, TABLE_TYPE, TABLE_SCHEMA, TABLE_ROWS, CREATE_TIME
   FROM INFORMATION_SCHEMA.TABLES
   WHERE TABLE_SCHEMA = "{database}"',
  {{user: $user, password: $password, host: $host, database: $database}}
)
YIELD row
UNWIND [row] AS r
MERGE (:Table {{name: r.TABLE_NAME}})
MERGE (:TableType {{name: r.TABLE_TYPE}})
MERGE (:Schema {{name: r.TABLE_SCHEMA}})
MERGE (:RowCount {{value: r.TABLE_ROWS}})
MERGE (:CreatedAt {{timestamp: r.CREATE_TIME}});
"""

QUERY_2 = """
CALL migrate.mysql(
  'SELECT TABLE_NAME, TABLE_TYPE, TABLE_SCHEMA, TABLE_ROWS, CREATE_TIME 
   FROM INFORMATION_SCHEMA.TABLES 
   WHERE TABLE_SCHEMA = "{database}"',
  {{user: $user, password: $password, host: $host, database: $database}}
)
YIELD row
UNWIND [row] AS r
WITH 
  r,
  coalesce(r.TABLE_ROWS, "NULL") AS rowCountVal
MERGE (t:Table {{name: r.TABLE_NAME}})
MERGE (tt:TableType {{name: r.TABLE_TYPE}})
MERGE (s:Schema {{name: r.TABLE_SCHEMA}})
MERGE (rc:RowCount {{value: rowCountVal}})
MERGE (c:CreatedAt {{timestamp: r.CREATE_TIME}})
MERGE (t)-[:HAS_TYPE]->(tt)
MERGE (t)-[:HAS_SCHEMA]->(s)
MERGE (t)-[:HAS_ROWCOUNT]->(rc)
MERGE (t)-[:HAS_CREATED_AT]->(c);
"""
QUERY_3A = """
CALL migrate.mysql(
  'SELECT kcu.TABLE_NAME, kcu.COLUMN_NAME, tc.CONSTRAINT_TYPE
   FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu
   JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
     ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME
    AND kcu.TABLE_NAME = tc.TABLE_NAME
   WHERE kcu.CONSTRAINT_SCHEMA = "{database}"',
  {{user: $user, password: $password, host: $host, database: $database}}
)
YIELD row
UNWIND [row] AS r
WITH 
  coalesce(r.TABLE_NAME, "NULL") AS tableName,
  coalesce(r.COLUMN_NAME, "NULL") AS columnName,
  coalesce(r.CONSTRAINT_TYPE, "NONE") AS constraintType

MERGE (t:Table {{name: tableName}})
MERGE (c:Column {{id: tableName + "." + columnName, name: columnName}})
MERGE (t)-[:HAS_COLUMN]->(c)

FOREACH (_ IN CASE WHEN constraintType = "PRIMARY KEY" THEN [1] ELSE [] END |
  MERGE (c)-[:IS_PRIMARY_KEY_OF]->(t)
)
FOREACH (_ IN CASE WHEN constraintType = "FOREIGN KEY" THEN [1] ELSE [] END |
  MERGE (c)-[:IS_FOREIGN_KEY_OF]->(t)
);
"""

QUERY_3B = """
CALL migrate.mysql(
  'SELECT CONSTRAINT_NAME, TABLE_NAME, COLUMN_NAME, 
          REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME 
   FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE 
   WHERE CONSTRAINT_SCHEMA = "{database}" AND REFERENCED_TABLE_NAME IS NOT NULL',
  {{user: $user, password: $password, host: $host, database: $database}}
)
YIELD row
UNWIND [row] AS r
WITH 
  coalesce(r.CONSTRAINT_NAME, "NULL") AS constraintName,
  coalesce(r.TABLE_NAME, "NULL") AS tableName,
  coalesce(r.COLUMN_NAME, "NULL") AS columnName,
  coalesce(r.REFERENCED_TABLE_NAME, "NULL") AS refTableName,
  coalesce(r.REFERENCED_COLUMN_NAME, "NULL") AS refColumnName


MERGE (srcTable:Table {{name: tableName}})
MERGE (refTable:Table {{name: refTableName}})
MERGE (constraint:Constraint {{name: constraintName}})
MERGE (srcCol:Column {{id: tableName + "." + columnName, name: columnName}})
MERGE (refCol:Column {{id: refTableName + "." + refColumnName, name: refColumnName}})

MERGE (srcTable)-[:REFERENCES {{via: constraintName}}]->(refTable)
MERGE (constraint)-[:USES_COLUMN]->(srcCol)
MERGE (constraint)-[:REFERENCES_COLUMN]->(refCol)
MERGE (srcCol)-[:REFERENCES_COLUMN]->(refCol)
MERGE (srcCol)-[:REFERENCES_TABLE]->(refTable);
"""

QUERY_4 = """
CALL migrate.mysql(
  'SELECT TABLE_NAME, COLUMN_NAME, COLUMN_TYPE, IS_NULLABLE 
   FROM INFORMATION_SCHEMA.COLUMNS 
   WHERE TABLE_SCHEMA = "{database}"',
  {{user: $user, password: $password, host: $host, database: $database}}
)
YIELD row
UNWIND [row] AS r
WITH
  coalesce(r.TABLE_NAME, "NULL") AS tableName,
  coalesce(r.COLUMN_NAME, "NULL") AS columnName,
  coalesce(r.COLUMN_TYPE, "UNKNOWN") AS columnType,
  coalesce(r.IS_NULLABLE, "UNKNOWN") AS isNullable

MERGE (c:Column {{id: tableName + "." + columnName, name: columnName}})

MERGE (t:ColumnType {{name: columnType}})
MERGE (n:Nullable {{name: isNullable, value: isNullable}})

MERGE (c)-[:HAS_TYPE]->(t)
MERGE (c)-[:IS_NULLABLE]->(n);
"""


class SqlConfig(BaseModel):
    user: str
    password: str
    host: str
    database: str


class SqlToMemERDMigration(MemgraphPropertyGraphStore):
    def __init__(
        self,
        graph_url: str = GRAPH_URL,
        graph_username: str = GRAPH_USER,
        graph_password: str = GRAPH_PASSWORD,
        sql_config: Optional[SqlConfig] = None,
    ):
        super().__init__(username=graph_username, password=graph_password, url=graph_url)
        self.sql_config = sql_config or SqlConfig(
            user=MYSQL_CONFIG["user"],
            password=MYSQL_CONFIG["password"],
            host=MYSQL_CONFIG["host"],
            database=MYSQL_CONFIG["database"],
        )

    def migrate_erd(self):
        print("Clearing existing graph...")
        self.structured_query(QUERY_RESET)

        params = self.sql_config.model_dump()

        db_name = params["database"]

        print("Running migration Query 1...")
        self.structured_query(QUERY_1.format(
            database=db_name), param_map=params)

        print("Running migration Query 2...")
        self.structured_query(QUERY_2.format(
            database=db_name), param_map=params)

        print("Running migration Query 3A...")
        self.structured_query(QUERY_3A.format(
            database=db_name), param_map=params)

        print("Running migration Query 3B...")
        self.structured_query(QUERY_3B.format(
            database=db_name), param_map=params)

        print("Running migration Query 4...")
        self.structured_query(QUERY_4.format(
            database=db_name), param_map=params)

        print("Migration complete.")
