# src/db_manager.py
"""Módulo de gestión de base de datos.

Maneja conexiones a SQLite asegurando prácticas seguras para subprocesos,
inyección de dependencias para rutas dinámicas y liberación de recursos.
"""

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DB_PATH


def execute_query(query: str, parameters: tuple = (), db_path: Optional[Path] = None) -> None:
    """Ejecuta una consulta SQL aislada garantizando el cierre seguro de la conexión.
    
    Args:
        query (str): Cadena de ejecución SQL.
        parameters (tuple, opcional): Parámetros de la consulta para prevenir 
            inyección SQL. Por defecto es ().
        db_path (Path, opcional): Ruta hacia el archivo SQLite. Si es None, 
            usa la ruta principal configurada por defecto.
    """
    target_path = db_path if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(target_path)
    try:
        cursor = conn.cursor()
        cursor.execute(query, parameters)
        conn.commit()
    finally:
        conn.close()


def save_dataframe_to_table(
    df: pd.DataFrame, 
    table_name: str, 
    if_exists: str = 'replace', 
    db_path: Optional[Path] = None
) -> None:
    """Persiste un DataFrame de pandas en la base de datos SQLite.
    
    Args:
        df (pd.DataFrame): Datos que serán almacenados.
        table_name (str): Nombre de la tabla de destino.
        if_exists (str, opcional): Comportamiento si la tabla ya existe 
            ('fail', 'replace', 'append'). Por defecto es 'replace'.
        db_path (Path, opcional): Ruta hacia el archivo SQLite de destino. 
            Si es None, usa la ruta principal configurada en src.config.
    """
    target_path = db_path if db_path else DB_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(target_path)
    try:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
    finally:
        conn.close()


def load_table_to_dataframe(table_name: str, db_path: Optional[Path] = None) -> Optional[pd.DataFrame]:
    """Extrae una tabla de la base de datos SQLite hacia un DataFrame de pandas.
    
    Args:
        table_name (str): Nombre de la tabla de destino a extraer.
        db_path (Path, opcional): Ruta hacia el archivo SQLite de origen. 
            Si es None, usa la ruta principal configurada por defecto.
        
    Returns:
        Optional[pd.DataFrame]: El DataFrame extraído, o None si ocurre un error 
            (ej. si la tabla o la base de datos origen no existen).
    """
    target_path = db_path if db_path else DB_PATH
    
    # Decisión técnica: Evita que SQLite cree un archivo vacío (fantasma) 
    # automáticamente si la ruta proporcionada es incorrecta o no existe aún.
    if not target_path.exists():
        print(f"Error: La base de datos origen no existe en {target_path}")
        return None
        
    conn = sqlite3.connect(target_path)
    try:
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        return df
    except Exception as e:
        print(f"Error de extracción en base de datos para la tabla '{table_name}': {str(e)}")
        return None
    finally:
        conn.close()

def log_execution_time(script_name: str, elapsed_seconds: float, subject_id: str = "GLOBAL") -> None:
    """Registra el tiempo de ejecución en la base de datos de auditoría.
    
    Actualiza o inserta el tiempo que tomó procesar un script (o un sujeto 
    dentro de un script) usando la función nativa UPSERT de SQLite.
    
    Args:
        script_name (str): Nombre del script o proceso.
        elapsed_seconds (float): Tiempo transcurrido en segundos.
        subject_id (str, opcional): ID del sujeto. Por defecto es "GLOBAL".
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_execution_times (
                script_name TEXT,
                subject_id TEXT,
                execution_time_seconds REAL,
                timestamp TEXT,
                PRIMARY KEY (script_name, subject_id)
            )
        ''')
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute('''
            INSERT OR REPLACE INTO audit_execution_times 
            (script_name, subject_id, execution_time_seconds, timestamp) 
            VALUES (?, ?, ?, ?)
        ''', (script_name, subject_id, elapsed_seconds, timestamp))
        conn.commit()
    except Exception as e:
        print(f"Error guardando tiempo de ejecución: {e}")
    finally:
        conn.close()
