-- Schema SQLite para ClimaSafeAI
-- sqlite3 data/climasafe.db < data/schema.sql

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ── Perfiles de usuario ───────────────────────────────────────────
CREATE TABLE IF NOT EXISTS perfiles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    alias           TEXT UNIQUE,                   -- nombre opcional para identificar
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now')),

    -- Campos escalares (todos opcionales)
    edad            INTEGER,
    fecha_nacimiento TEXT,                           -- YYYY-MM-DD
    sexo            TEXT CHECK(sexo IN ('hombre', 'mujer')),
    porcentaje_grasa REAL,
    nivel_actividad TEXT CHECK(nivel_actividad IN ('reposo', 'ligera', 'moderada', 'intensa', 'muy_intensa')),
    hora_inicio     REAL CHECK(hora_inicio IS NULL OR (hora_inicio >= 0 AND hora_inicio < 24)),
    duracion_actividad_h REAL,
    aclimatado      INTEGER,                       -- 0/1 (SQLite bool)
    aclimatado_actualizado_en TEXT,                 -- timestamp del último cambio en aclimatado (para auto-aclimatación)
    falta_sueno     INTEGER,
    enfermedad_reciente INTEGER,
    alcohol_reciente INTEGER,
    fototipo        INTEGER CHECK(fototipo IS NULL OR (fototipo >= 1 AND fototipo <= 6)),
    entrenado       TEXT CHECK(entrenado IN ('si', 'no')),
    deporte         TEXT,
    ocupacion       TEXT CHECK(ocupacion IN ('oficina', 'reparto', 'mantenimiento', 'construccion', 'campo')),
    fiesta          INTEGER,                       -- 0/1
    lat             REAL,
    lon             REAL,
    provincia       TEXT,
    tags            TEXT,                            -- coma-separadas: electricista,fontanero
    telegram_chat_id TEXT UNIQUE,                    -- chat_id de Telegram para vincular conversación

    -- Control de acceso del MCP (MCP-003)
    uid             TEXT,                            -- id público opaco 'usr_...'; unicidad vía idx_perfiles_uid
    mcp_token_hash  TEXT,                            -- sha256 del secreto del llamante; NULL = sin acceso MCP
    rol             TEXT NOT NULL DEFAULT 'usuario'  -- 'usuario' | 'admin'
);
CREATE INDEX IF NOT EXISTS idx_perfiles_alias ON perfiles(alias);
CREATE INDEX IF NOT EXISTS idx_perfiles_created ON perfiles(created_at);
-- El índice idx_perfiles_telegram se crea en _migrate() tras añadir la columna
-- Los de uid y mcp_token_hash también, para que BD nuevas y migradas coincidan

-- ── Relaciones muchos-a-muchos ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS perfil_comorbilidades (
    perfil_id INTEGER NOT NULL REFERENCES perfiles(id) ON DELETE CASCADE,
    clave     TEXT NOT NULL,                        -- ej: "cardiovascular", "diabetes"
    PRIMARY KEY (perfil_id, clave)
);

CREATE TABLE IF NOT EXISTS perfil_farmacos (
    perfil_id INTEGER NOT NULL REFERENCES perfiles(id) ON DELETE CASCADE,
    clave     TEXT NOT NULL,                        -- ej: "antipsicoticos", "diureticos_asa"
    PRIMARY KEY (perfil_id, clave)
);

CREATE TABLE IF NOT EXISTS perfil_situacion_social (
    perfil_id INTEGER NOT NULL REFERENCES perfiles(id) ON DELETE CASCADE,
    clave     TEXT NOT NULL,                        -- ej: "vive_solo", "vivienda_fria"
    PRIMARY KEY (perfil_id, clave)
);

CREATE TABLE IF NOT EXISTS perfil_ocupacional (
    perfil_id INTEGER NOT NULL REFERENCES perfiles(id) ON DELETE CASCADE,
    clave     TEXT NOT NULL,                        -- ej: "estres_termico_laboral"
    PRIMARY KEY (perfil_id, clave)
);

-- ── Factores de riesgo (migrado desde JSON) ────────────────────────
CREATE TABLE IF NOT EXISTS factores_riesgo (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tipo        TEXT NOT NULL CHECK(tipo IN ('calor', 'frio')),
    categoria   TEXT NOT NULL,                      -- comorbilidades, farmacos, fisiologico, situacional, ocupacional
    clave       TEXT NOT NULL,                      -- ej: "cardiovascular", "no_aclimatado"
    nombre      TEXT NOT NULL,                      -- nombre humano
    coef        REAL NOT NULL CHECK(coef > 0),
    doi         TEXT,
    calidad     TEXT NOT NULL DEFAULT 'baja' CHECK(calidad IN ('alta', 'media', 'baja')),
    poblacion   TEXT,                               -- opcional, texto libre
    implementado INTEGER NOT NULL DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(tipo, categoria, clave)
);

CREATE INDEX IF NOT EXISTS idx_factores_tipo ON factores_riesgo(tipo);
CREATE INDEX IF NOT EXISTS idx_factores_implementado ON factores_riesgo(implementado);

-- ── Historial de consultas ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS historial_consultas (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    perfil_id       INTEGER REFERENCES perfiles(id) ON DELETE SET NULL,
    provincia       TEXT NOT NULL DEFAULT 'Madrid',
    lat             REAL,
    lon             REAL,
    tipo_riesgo     TEXT CHECK(tipo_riesgo IN ('calor', 'frio')),
    indice_original REAL,
    indice_personalizado REAL,
    clase_final     TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_historial_perfil ON historial_consultas(perfil_id);
CREATE INDEX IF NOT EXISTS idx_historial_fecha ON historial_consultas(created_at);

-- ── Tags predefinidas ────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tags_disponibles (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    nombre  TEXT NOT NULL UNIQUE
);

-- ── Aprendizaje activo (scout) ──────────────────────────────────────
CREATE TABLE IF NOT EXISTS scout_entrenamiento (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    titulo       TEXT NOT NULL,
    abstract     TEXT NOT NULL,
    embedding    BLOB,
    veredicto    TEXT NOT NULL CHECK(veredicto IN ('aceptable', 'irrelevante')),
    fuente       TEXT NOT NULL DEFAULT 'llm',
    created_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Rutinas semanales (BOT-007) ───────────────────────────────────
CREATE TABLE IF NOT EXISTS rutinas (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     TEXT NOT NULL,                -- chat_id de Telegram
    nombre      TEXT NOT NULL,                -- ej: "trabajo", "entreno"
    dias        TEXT NOT NULL,                -- coma-separada 1-7, 1=lunes
    hora_inicio REAL NOT NULL,
    hora_fin    REAL NOT NULL,
    ocupacion   TEXT,                          -- si la rutina es laboral
    deporte     TEXT,                          -- si la rutina es deportiva
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_rutinas_chat ON rutinas(chat_id);

-- ── Hora de aviso diario por chat (BOT-007) ───────────────────────
CREATE TABLE IF NOT EXISTS avisos_config (
    chat_id     TEXT PRIMARY KEY,
    hora        TEXT NOT NULL,                 -- 'HH:MM'
    updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Las tablas vec0 (factores_vec, factores_vec_src) se crean desde
-- RAG.initialize() con sqlite-vec cargado. No incluirlas aquí
-- porque DBManager.initialize() ejecuta este script sin la extensión.
