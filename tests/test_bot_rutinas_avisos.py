"""Tests de BOT-007: /perfil editable, /rutinas y avisos diarios.

Sigue el patrón de tests/test_telegram_bot.py: el bot no toca el SQLite real,
se monkeypatchea `_db` con fakes y `_tg` con un stub que captura mensajes.
Los métodos de DBManager (rutinas y avisos) sí se prueban contra un SQLite
temporal en tmp_path.
"""

from __future__ import annotations

from datetime import date

import pytest

from climasafeai.bot.telegram_bot import (
    Estado,
    _conversaciones,
)
from climasafeai.db.manager import DBManager


class _HoyFijo(date):
    """Fija 'hoy' en 2026-08-03 para que la edad calculada sea determinista."""

    @classmethod
    def today(cls):
        return cls(2026, 8, 3)


@pytest.fixture(autouse=True)
def limpiar():
    _conversaciones.clear()
    yield
    _conversaciones.clear()


def _sin_modelo(monkeypatch):
    import climasafeai.bot.telegram_bot as mod

    monkeypatch.setattr(mod, "_modelo_por_defecto", lambda: mod.MODELO_DETERMINISTA)


def _stub_tg(monkeypatch):
    """Sustituye _tg por un stub que captura los mensajes enviados."""
    import climasafeai.bot.telegram_bot as mod

    enviados: list[str] = []

    async def _fake_tg(method: str, **kwargs):
        if method == "sendMessage":
            enviados.append(kwargs.get("text", ""))
        return {"ok": True, "result": {}}

    monkeypatch.setattr(mod, "_tg", _fake_tg)
    return enviados


# ── DBManager: rutinas y avisos (SQLite real en tmp_path) ───────────────────


class TestDBRutinas:
    def test_crear_listar_y_borrar(self, tmp_path):
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        rid = db.crear_rutina("1", "trabajo", "1,2,3,4,5", 8.0, 16.0)
        rid2 = db.crear_rutina("1", "correr", "6,7", 18.0, 20.0, deporte="correr")
        assert rid2 > rid

        rutinas = db.listar_rutinas("1")
        assert len(rutinas) == 2
        assert rutinas[0]["hora_inicio"] == 8.0  # ordenado por hora
        assert rutinas[1]["deporte"] == "correr"
        assert rutinas[0]["dias"] == "1,2,3,4,5"
        # Los chats no se mezclan
        assert db.listar_rutinas("999") == []

        assert db.eliminar_rutina(rid)
        assert len(db.listar_rutinas("1")) == 1

    def test_rutinas_por_dia(self, tmp_path):
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        rid = db.crear_rutina("1", "trabajo", "1,2,3,4,5", 8.0, 16.0)
        rid2 = db.crear_rutina("1", "correr", "6,7", 18.0, 20.0)
        # 1=lunes, 3=miércoles (cae en L-V), 6=sábado, 7=domingo (cae en S-D)
        assert [r["id"] for r in db.rutinas_por_dia("1", 1)] == [rid]
        assert [r["id"] for r in db.rutinas_por_dia("1", 3)] == [rid]
        assert [r["id"] for r in db.rutinas_por_dia("1", 6)] == [rid2]
        assert [r["id"] for r in db.rutinas_por_dia("1", 7)] == [rid2]
        assert db.rutinas_por_dia("999", 1) == []


class TestDBAvisos:
    def test_guardar_obtener_y_desactivar(self, tmp_path):
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        assert db.obtener_hora_aviso("1") is None

        db.guardar_hora_aviso("1", "08:00")
        assert db.obtener_hora_aviso("1") == "08:00"
        db.guardar_hora_aviso("1", "09:30")
        assert db.obtener_hora_aviso("1") == "09:30"
        assert db.chats_con_aviso() == [{"chat_id": "1", "hora": "09:30"}]

        db.guardar_hora_aviso("1", None)
        assert db.obtener_hora_aviso("1") is None
        assert db.chats_con_aviso() == []


class TestDBUltimaSalida:
    """BOT-017: la última salida se guarda como JSON en el perfil y se lee
    como dict, para que /start pueda ofrecer repetirla."""

    def test_guardar_y_recuperar_ultima_salida(self, tmp_path):
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        pid = db.crear_perfil({"alias": "Aldán", "telegram_chat_id": "1"})
        salida = {
            "actividad": "Correr", "nivel_actividad": "muy_intensa",
            "duracion_h": 2.0, "hora_inicio": 8,
            "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra",
            "entrenado": True, "clase_final": 1,
        }
        db.actualizar_perfil(pid, {"ultima_salida": salida})
        perfil = db.obtener_perfil(pid)
        assert perfil["ultima_salida"] == salida

    def test_sin_ultima_salida_devuelve_none(self, tmp_path):
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        pid = db.crear_perfil({"alias": "Aldán", "telegram_chat_id": "1"})
        perfil = db.obtener_perfil(pid)
        assert perfil.get("ultima_salida") is None

    def test_ultima_salida_se_sobreescribe(self, tmp_path):
        """Cada /start guarda la última salida: la anterior se reemplaza."""
        db = DBManager(tmp_path / "test.db")
        db.initialize()
        pid = db.crear_perfil({"alias": "Aldán", "telegram_chat_id": "1"})
        db.actualizar_perfil(pid, {"ultima_salida": {"actividad": "Correr", "hora_inicio": 8}})
        db.actualizar_perfil(pid, {"ultima_salida": {"actividad": "Senderismo", "hora_inicio": 9}})
        perfil = db.obtener_perfil(pid)
        assert perfil["ultima_salida"] == {"actividad": "Senderismo", "hora_inicio": 9}


# ── Parsing y formateo de rutinas ───────────────────────────────────────────


class TestParsearRutina:
    def test_lv_trabajo_8_16(self):
        from climasafeai.bot.telegram_bot import _parsear_rutina

        r = _parsear_rutina("L-V trabajo 8-16")
        assert r["dias"] == "1,2,3,4,5"
        assert r["nombre"] == "trabajo"
        assert r["hora_inicio"] == 8.0
        assert r["hora_fin"] == 16.0
        assert r["deporte"] is None

    def test_lv_entreno_18_20(self):
        from climasafeai.bot.telegram_bot import _parsear_rutina

        r = _parsear_rutina("L-V entreno 18-20")
        assert r["dias"] == "1,2,3,4,5"
        assert r["nombre"] == "entreno"
        assert r["hora_inicio"] == 18.0
        assert r["hora_fin"] == 20.0

    def test_dias_sueltos_y_deporte_conocido(self):
        from climasafeai.bot.telegram_bot import _parsear_rutina

        r = _parsear_rutina("L,X correr 18-20")
        assert r["dias"] == "1,3"
        assert r["deporte"] == "correr"

    def test_horas_con_minutos(self):
        from climasafeai.bot.telegram_bot import _parsear_rutina

        r = _parsear_rutina("L-V trabajo 8:30-14:00")
        assert r["hora_inicio"] == 8.5
        assert r["hora_fin"] == 14.0

    def test_invalidas(self):
        from climasafeai.bot.telegram_bot import _parsear_rutina

        assert _parsear_rutina("hola") is None
        assert _parsear_rutina("L-V trabajo") is None  # sin horas
        assert _parsear_rutina("L-V trabajo 20-8") is None  # fin < inicio
        assert _parsear_rutina("Z-V trabajo 8-16") is None  # día malo
        assert _parsear_rutina("L-V trabajo 25-16") is None  # hora mal

    def test_formatear_dias(self):
        from climasafeai.bot.telegram_bot import _formatear_dias

        assert _formatear_dias("1,2,3,4,5") == "L-V"
        assert _formatear_dias("1,3,5") == "L,X,V"
        assert _formatear_dias("6,7") == "S-D"

    def test_formato_hora(self):
        from climasafeai.bot.telegram_bot import _formato_hora

        assert _formato_hora(8.0) == "8:00"
        assert _formato_hora(18.5) == "18:30"

    def test_validar_hora(self):
        from climasafeai.bot.telegram_bot import _validar_hora

        assert _validar_hora("08:00") == "08:00"
        assert _validar_hora("8:05") == "08:05"
        assert _validar_hora("25:00") is None
        assert _validar_hora("8") is None
        assert _validar_hora("abc") is None


# ── /perfil: mostrar y editar ───────────────────────────────────────────────


class _FakeDBPerfil:
    """Perfil guardado + registro de actualizaciones."""

    def __init__(self, perfil=None):
        self.actualizaciones: list[dict] = []
        self.perfil = perfil or {
            "id": 7,
            "alias": "Aldán",
            "edad": 57,
            "sexo": "hombre",
            "porcentaje_grasa": 20.5,
            "fototipo": 3,
            "aclimatado": False,
            "comorbilidades": ["cardiovascular"],
            "farmacos": ["diureticos_asa"],
            "situacion_social": ["vive_solo"],
            "lat": 42.29,
            "lon": -8.81,
            "provincia": "Pontevedra",
        }

    def buscar_por_telegram(self, chat_id):
        return {"id": 7, "alias": "Aldán"} if chat_id == "1" else None

    def obtener_perfil(self, _pid):
        return dict(self.perfil)

    def actualizar_perfil(self, pid, datos):
        self.actualizaciones.append(dict(datos))
        for k, v in datos.items():
            self.perfil[k] = v
        return True


class _FakeDBSinPerfil:
    def buscar_por_telegram(self, chat_id):
        return None


class TestPerfil:
    @pytest.mark.asyncio
    async def test_perfil_muestra_datos_actuales(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "_db", _FakeDBPerfil())
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/perfil")

        assert "Aldán" in r
        assert "57" in r and "hombre" in r
        assert "20.5" in r
        assert "cardiovascular" in r
        assert "diureticos_asa" in r
        assert "vive_solo" in r
        assert "Pontevedra" in r

    @pytest.mark.asyncio
    async def test_perfil_sin_perfil(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "_db", _FakeDBSinPerfil())
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/perfil")

        assert "no tienes perfil" in r.lower()

    @pytest.mark.asyncio
    async def test_editar_edad_por_texto(self, monkeypatch):
        """BOT-010: editar edad pide fecha de nacimiento y guarda la edad calculada."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "date", _HoyFijo)
        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r, _ = await mod.procesar_callback(1, "edit_edad")
        assert r == mod._PREGUNTAS_EDIT["edad"]
        assert "fecha de nacimiento" in r.lower()
        assert _conversaciones[1]["_editando"] == "edad"

        r2 = await mod.procesar_mensaje(1, "15/03/1990")
        assert "Edad" in r2 and "actualizado" in r2
        assert db.actualizaciones == [{"edad": 36}]
        assert "_editando" not in _conversaciones[1]

    @pytest.mark.asyncio
    async def test_editar_edad_fecha_futura_no_guarda(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "date", _HoyFijo)
        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_edad")
        r = await mod.procesar_mensaje(1, "15/03/2030")

        assert "futuro" in r
        assert db.actualizaciones == []

    @pytest.mark.asyncio
    async def test_editar_grasa_por_texto(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_grasa")
        r = await mod.procesar_mensaje(1, "22,5%")

        assert "grasa" in r.lower()
        assert db.actualizaciones == [{"porcentaje_grasa": 22.5}]  # columna real

    @pytest.mark.asyncio
    async def test_editar_sexo_con_boton(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_sexo")
        r, _ = await mod.procesar_callback(1, "mujer")

        assert "Sexo" in r and "actualizado" in r
        assert db.actualizaciones == [{"sexo": "mujer"}]

    @pytest.mark.asyncio
    async def test_editar_aclimatado_con_boton(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_aclimatado")
        await mod.procesar_callback(1, "si")

        assert db.actualizaciones == [{"aclimatado": True}]

    @pytest.mark.asyncio
    async def test_editar_fototipo_con_boton(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_fototipo")
        await mod.procesar_callback(1, "5")

        assert db.actualizaciones == [{"fototipo": 5}]

    @pytest.mark.asyncio
    async def test_editar_comorbilidades_multiselect(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_comorbilidades")
        await mod.procesar_callback(1, "cardiovascular")
        await mod.procesar_callback(1, "diabetes")
        r, _ = await mod.procesar_callback(1, "__done__")

        assert "Comorbilidades" in r
        assert db.actualizaciones == [{"comorbilidades": {"cardiovascular", "diabetes"}}]

    @pytest.mark.asyncio
    async def test_editar_medicacion_guarda_farmacos(self, monkeypatch):
        """El campo se llama 'medicacion' pero la columna real es `farmacos`."""
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_medicacion")
        await mod.procesar_callback(1, "diureticos_asa")
        await mod.procesar_callback(1, "__done__")

        assert db.actualizaciones == [{"farmacos": {"diureticos_asa"}}]

    @pytest.mark.asyncio
    async def test_editar_situacion_social_por_texto(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBPerfil()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_situacion_social")
        r = await mod.procesar_mensaje(1, "vive_solo, sin_aire_acondicionado")

        assert "Situación social" in r
        assert db.actualizaciones == [{"situacion_social": ["vive_solo", "sin_aire_acondicionado"]}]

    @pytest.mark.asyncio
    async def test_editar_sin_perfil_no_guarda(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "date", _HoyFijo)
        monkeypatch.setattr(mod, "_db", _FakeDBSinPerfil())
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_callback(1, "edit_edad")
        r = await mod.procesar_mensaje(1, "15/03/1990")

        assert "no hay perfil" in r.lower()


# ── /rutinas: listar, añadir y borrar ───────────────────────────────────────


class _FakeDBRutinas:
    def __init__(self):
        self.rutinas: list[dict] = []
        self.eliminados: list[int] = []

    def buscar_por_telegram(self, chat_id):
        return None

    def listar_rutinas(self, chat_id):
        return [dict(r) for r in self.rutinas]

    def crear_rutina(self, chat_id, **datos):
        rid = len(self.rutinas) + 1
        self.rutinas.append({"id": rid, "chat_id": str(chat_id), **datos})
        return rid

    def eliminar_rutina(self, rid):
        self.eliminados.append(rid)
        self.rutinas = [r for r in self.rutinas if r["id"] != rid]
        return True


class TestRutinas:
    @pytest.mark.asyncio
    async def test_sin_rutinas_explica_como_anadir(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        monkeypatch.setattr(mod, "_db", _FakeDBRutinas())
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas")

        assert "no tienes rutinas" in r.lower()
        assert "/rutinas_anadir" in r

    @pytest.mark.asyncio
    async def test_anadir_rutina_trabajo_pregunta_tipo_y_no_guarda(self, monkeypatch):
        """BOT-015: una rutina de trabajo no se guarda directo, pregunta el tipo."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas_anadir L-V trabajo 8-16")

        assert "tipo de trabajo" in r.lower()
        assert db.rutinas == []  # no guarda directo
        pendiente = _conversaciones[1]["_rutina_pendiente"]
        assert pendiente["dias"] == "1,2,3,4,5"
        assert pendiente["hora_inicio"] == 8.0
        assert pendiente["hora_fin"] == 16.0
        assert pendiente["nombre"] == "trabajo"
        assert "ocupacion" not in pendiente

    @pytest.mark.asyncio
    async def test_anadir_rutina_trabajo_guarda_con_ocupacion_al_elegir_tipo(self, monkeypatch):
        """BOT-015: elegir el tipo de trabajo guarda la rutina con su ocupación."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L-V trabajo 8-16")
        r, _ = await mod.procesar_callback(1, "rutina_tipo_construccion")

        assert len(db.rutinas) == 1
        assert db.rutinas[0]["ocupacion"] == "construccion"
        assert db.rutinas[0]["dias"] == "1,2,3,4,5"
        assert db.rutinas[0]["hora_inicio"] == 8.0
        assert "añadida" in r.lower()
        assert "Construcción x2.2" in r
        assert "_rutina_pendiente" not in _conversaciones[1]

    @pytest.mark.asyncio
    async def test_anadir_rutina_trabajo_oficina(self, monkeypatch):
        """BOT-015: 'oficina' es un tipo de trabajo válido (x1.0) en el cuestionario."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L-V trabajo 8-16")
        r, _ = await mod.procesar_callback(1, "rutina_tipo_oficina")

        assert db.rutinas[0]["ocupacion"] == "oficina"
        assert "Oficina / interior x1.0" in r

    @pytest.mark.asyncio
    async def test_cuestionario_trabajo_sin_pendiente_no_guarda(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r, _ = await mod.procesar_callback(1, "rutina_tipo_campo")

        assert "ninguna rutina pendiente" in r.lower()
        assert db.rutinas == []

    @pytest.mark.asyncio
    async def test_cuestionario_trabajo_tipo_invalido(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L-V trabajo 8-16")
        r, _ = await mod.procesar_callback(1, "rutina_tipo_buceo")

        assert "inválido" in r.lower()
        assert db.rutinas == []

    def test_kb_tipo_trabajo_incluye_oficina_con_prefijo(self):
        from climasafeai.bot.telegram_bot import _kb_tipo_trabajo

        kb = _kb_tipo_trabajo("rutina_tipo_")
        datas = [b["callback_data"] for fila in kb for b in fila]

        assert "rutina_tipo_oficina" in datas
        assert "rutina_tipo_campo" in datas
        assert all(d.startswith("rutina_tipo_") for d in datas)

    @pytest.mark.asyncio
    async def test_anadir_rutina_entreno_pregunta_tipo_de_actividad(self, monkeypatch):
        """BOT-016: 'entreno' no se guarda directo, pregunta la actividad."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas_anadir L-V entreno 18-20")

        assert "tipo de actividad" in r.lower()
        assert db.rutinas == []  # no guarda directo
        pendiente = _conversaciones[1]["_rutina_pendiente"]
        assert pendiente["hora_inicio"] == 18.0
        assert pendiente["hora_fin"] == 20.0
        assert pendiente["nombre"] == "entreno"
        assert pendiente.get("deporte") is None

    @pytest.mark.asyncio
    async def test_anadir_rutina_entreno_guarda_con_deporte_al_elegir(self, monkeypatch):
        """BOT-016: elegir la actividad guarda el entreno con su deporte."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L-V entreno 18-20")
        r, _ = await mod.procesar_callback(1, "rutina_deporte_correr")

        assert len(db.rutinas) == 1
        assert db.rutinas[0]["deporte"] == "correr"
        assert db.rutinas[0]["nombre"] == "entreno"
        assert db.rutinas[0].get("ocupacion") is None
        assert "añadida" in r.lower()
        assert "Correr MET 10.5" in r
        assert "_rutina_pendiente" not in _conversaciones[1]

    @pytest.mark.asyncio
    async def test_anadir_rutina_deporte_pregunta_tipo_y_no_guarda(self, monkeypatch):
        """BOT-016: un deporte de la lista DEPORTES no se guarda directo."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas_anadir L,X futbol 18-20")

        assert "tipo de actividad" in r.lower()
        assert db.rutinas == []  # no guarda directo
        pendiente = _conversaciones[1]["_rutina_pendiente"]
        assert pendiente["dias"] == "1,3"
        assert pendiente["nombre"] == "futbol"
        assert pendiente["deporte"] == "futbol"
        assert "ocupacion" not in pendiente

    @pytest.mark.asyncio
    async def test_anadir_rutina_deporte_guarda_con_deporte_al_elegir(self, monkeypatch):
        """BOT-016: elegir la actividad guarda el deporte con su etiqueta y MET."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L,X futbol 18-20")
        r, _ = await mod.procesar_callback(1, "rutina_deporte_futbol")

        assert len(db.rutinas) == 1
        assert db.rutinas[0]["deporte"] == "futbol"
        assert db.rutinas[0]["dias"] == "1,3"
        assert db.rutinas[0].get("ocupacion") is None
        assert "añadida" in r.lower()
        assert "Futbol MET 7" in r
        assert "_rutina_pendiente" not in _conversaciones[1]

    @pytest.mark.asyncio
    async def test_anadir_rutina_deporte_otro_met_cambia_la_etiqueta(self, monkeypatch):
        """BOT-016: futbol_competicion (9 MET) es otra actividad, no el mismo."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L,X futbol 18-20")
        r, _ = await mod.procesar_callback(1, "rutina_deporte_futbol_competicion")

        assert db.rutinas[0]["deporte"] == "futbol_competicion"
        assert "Futbol de competicion MET 9" in r

    @pytest.mark.asyncio
    async def test_cuestionario_deporte_sin_pendiente_no_guarda(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r, _ = await mod.procesar_callback(1, "rutina_deporte_correr")

        assert "ninguna rutina pendiente" in r.lower()
        assert db.rutinas == []

    @pytest.mark.asyncio
    async def test_cuestionario_deporte_tipo_invalido(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        await mod.procesar_mensaje(1, "/rutinas_anadir L-X correr 18-20")
        r, _ = await mod.procesar_callback(1, "rutina_deporte_padel")

        assert "inválida" in r.lower()
        assert db.rutinas == []

    def test_kb_tipo_deporte_incluye_deportes_con_prefijo(self):
        from climasafeai.bot.telegram_bot import _kb_tipo_deporte

        kb = _kb_tipo_deporte("rutina_deporte_")
        datas = [b["callback_data"] for fila in kb for b in fila]

        assert "rutina_deporte_futbol" in datas
        assert "rutina_deporte_correr" in datas
        assert all(d.startswith("rutina_deporte_") for d in datas)

    @pytest.mark.asyncio
    async def test_anadir_rutina_formato_invalido(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas_anadir L-V trabajo")

        assert "no entendí" in r.lower()
        assert db.rutinas == []

    @pytest.mark.asyncio
    async def test_listar_muestra_dias_y_horas(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        db.rutinas = [
            {
                "id": 1,
                "chat_id": "1",
                "nombre": "trabajo",
                "dias": "1,2,3,4,5",
                "hora_inicio": 8.0,
                "hora_fin": 16.0,
                "ocupacion": None,
                "deporte": None,
            },
            {
                "id": 2,
                "chat_id": "1",
                "nombre": "entreno",
                "dias": "6,7",
                "hora_inicio": 18.0,
                "hora_fin": 20.0,
                "ocupacion": None,
                "deporte": "correr",
            },
        ]
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas")

        assert "Trabajo" in r and "L-V, 8:00-16:00" in r
        assert "Entreno" in r and "S-D, 18:00-20:00" in r

    @pytest.mark.asyncio
    async def test_listar_muestra_ocupacion_con_intensidad(self, monkeypatch):
        """BOT-015: el resumen muestra la etiqueta del trabajo y su intensidad."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        db.rutinas = [
            {
                "id": 1,
                "chat_id": "1",
                "nombre": "trabajo",
                "dias": "1,2,3,4,5",
                "hora_inicio": 8.0,
                "hora_fin": 16.0,
                "ocupacion": "construccion",
                "deporte": None,
            },
        ]
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas")

        assert "Trabajo — L-V, 8:00-16:00 (Construcción x2.2)" in r

    @pytest.mark.asyncio
    async def test_listar_muestra_deporte_con_intensidad(self, monkeypatch):
        """BOT-016: el resumen muestra la etiqueta del deporte y su MET."""
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBRutinas()
        db.rutinas = [
            {
                "id": 1,
                "chat_id": "1",
                "nombre": "futbol",
                "dias": "1,3",
                "hora_inicio": 18.0,
                "hora_fin": 20.0,
                "ocupacion": None,
                "deporte": "futbol",
            },
        ]
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/rutinas")

        assert "Futbol — L,X, 18:00-20:00 (Futbol MET 7)" in r

    @pytest.mark.asyncio
    async def test_borrar_rutina_por_callback(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        db = _FakeDBRutinas()
        db.rutinas = [
            {
                "id": 3,
                "chat_id": "1",
                "nombre": "trabajo",
                "dias": "1,2,3,4,5",
                "hora_inicio": 8.0,
                "hora_fin": 16.0,
                "ocupacion": None,
                "deporte": None,
            },
        ]
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r, _ = await mod.procesar_callback(1, "del_rutina_3")

        assert "eliminada" in r.lower()
        assert db.eliminados == [3]
        assert db.rutinas == []


# ── /avisos: hora de aviso configurable ─────────────────────────────────────


class _FakeDBAvisos:
    def __init__(self):
        self.avisos: dict[str, str] = {}

    def buscar_por_telegram(self, chat_id):
        return None

    def obtener_hora_aviso(self, chat_id):
        return self.avisos.get(str(chat_id))

    def guardar_hora_aviso(self, chat_id, hora):
        if hora is None:
            self.avisos.pop(str(chat_id), None)
        else:
            self.avisos[str(chat_id)] = hora


class TestAvisos:
    @pytest.mark.asyncio
    async def test_configurar_hora(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBAvisos()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/avisos 08:00")

        assert "08:00" in r
        assert db.avisos == {"1": "08:00"}

    @pytest.mark.asyncio
    async def test_consultar_hora_actual(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBAvisos()
        db.avisos["1"] = "09:30"
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/avisos")

        assert "09:30" in r

    @pytest.mark.asyncio
    async def test_desactivar(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBAvisos()
        db.avisos["1"] = "08:00"
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/avisos off")

        assert "desactivado" in r.lower()
        assert db.avisos == {}

    @pytest.mark.asyncio
    async def test_hora_invalida(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        _sin_modelo(monkeypatch)
        db = _FakeDBAvisos()
        monkeypatch.setattr(mod, "_db", db)
        _conversaciones[1] = {"estado": Estado.IDLE, "data": {}}

        r = await mod.procesar_mensaje(1, "/avisos 25:99")

        assert "inválido" in r.lower()
        assert db.avisos == {}


# ── Aviso diario: calcula riesgo por ventana de rutina ──────────────────────


class TestAvisoDiario:
    @pytest.mark.asyncio
    async def test_sin_perfil_avisa_de_configurar_y_no_calcula(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        monkeypatch.setattr(mod, "_db", _FakeDBSinPerfil())
        enviados = _stub_tg(monkeypatch)

        def _no_deberia_calcular(**kwargs):
            raise AssertionError("no se calcula sin perfil")

        monkeypatch.setattr(mod, "predict_ensemble", _no_deberia_calcular)

        await mod._enviar_aviso_diario("1", 1)

        assert len(enviados) == 1
        assert "perfil" in enviados[0].lower()

    @pytest.mark.asyncio
    async def test_sin_ubicacion_avisa_y_no_calcula(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        class _DB:
            def buscar_por_telegram(self, chat_id):
                return {"id": 7} if chat_id == "1" else None

            def obtener_perfil(self, _pid):
                return {
                    "id": 7,
                    "edad": 57,
                    "sexo": "hombre",
                    "lat": None,
                    "lon": None,
                    "provincia": None,
                }

        monkeypatch.setattr(mod, "_db", _DB())
        enviados = _stub_tg(monkeypatch)

        def _no_deberia_calcular(**kwargs):
            raise AssertionError("no se calcula sin ubicación")

        monkeypatch.setattr(mod, "predict_ensemble", _no_deberia_calcular)

        await mod._enviar_aviso_diario("1", 1)

        assert len(enviados) == 1
        assert "ubicación" in enviados[0] or "ubicacion" in enviados[0].lower()

    @pytest.mark.asyncio
    async def test_con_perfil_y_rutina_llama_predict_con_la_ventana(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        class _DB:
            def buscar_por_telegram(self, chat_id):
                return {"id": 7} if chat_id == "1" else None

            def obtener_perfil(self, _pid):
                return {
                    "id": 7,
                    "alias": "Aldán",
                    "edad": 57,
                    "sexo": "hombre",
                    "porcentaje_grasa": 20.5,
                    "fototipo": 3,
                    "aclimatado": False,
                    "comorbilidades": [],
                    "farmacos": [],
                    "situacion_social": [],
                    "lat": 42.29,
                    "lon": -8.81,
                    "provincia": "Pontevedra",
                }

            def rutinas_por_dia(self, chat_id, weekday):
                if weekday == 1:
                    return [
                        {
                            "id": 1,
                            "nombre": "trabajo",
                            "dias": "1,2,3,4,5",
                            "hora_inicio": 8.0,
                            "hora_fin": 16.0,
                            "ocupacion": None,
                            "deporte": None,
                        }
                    ]
                return []

        capturado: dict = {}

        def _fake_predict(**kwargs):
            capturado.update(kwargs)
            return {
                "clase_final_label": "PRECAUCIÓN",
                "perfil": {"calor": {"prob_personalizada": 0.35}},
                "weather": {
                    "perfil_horario": [
                        {"hora": 8, "HI": 27.0, "temp": 28.0},
                        {"hora": 9, "HI": 28.0, "temp": 29.0},
                    ],
                    "current": {"t2m_c": 28.0, "rh": 60},
                },
                "modelos": {
                    "Formula": {"frio": {"wind_chill_c": 20}, "calor": {"heat_index_c": 28}}
                },
            }

        monkeypatch.setattr(mod, "_db", _DB())
        monkeypatch.setattr(mod, "predict_ensemble", _fake_predict)
        enviados = _stub_tg(monkeypatch)

        await mod._enviar_aviso_diario("1", 1)

        # La ventana a evaluar es la de la rutina, no la del perfil
        perfil_pred = capturado["perfil"]
        assert capturado["lat"] == 42.29 and capturado["lon"] == -8.81
        assert capturado["provincia"] == "Pontevedra"
        assert perfil_pred["hora_inicio"] == 8.0
        assert perfil_pred["duracion_actividad_h"] == 8.0

        assert len(enviados) == 1
        assert "Trabajo 8:00-16:00" in enviados[0]
        assert "PRECAUCIÓN" in enviados[0]
        assert "35%" in enviados[0]

    @pytest.mark.asyncio
    async def test_aviso_usa_ocupacion_de_la_rutina(self, monkeypatch):
        """BOT-015: el aviso pasa la ocupación de la rutina a la predicción y
        la etiqueta muestra su intensidad, con la ubicación del perfil del día."""
        import climasafeai.bot.telegram_bot as mod

        class _DB:
            def buscar_por_telegram(self, chat_id):
                return {"id": 7} if chat_id == "1" else None

            def obtener_perfil(self, _pid):
                return {
                    "id": 7,
                    "alias": "Aldán",
                    "edad": 57,
                    "sexo": "hombre",
                    "aclimatado": False,
                    "lat": 42.29,
                    "lon": -8.81,
                    "provincia": "Pontevedra",
                }

            def rutinas_por_dia(self, chat_id, weekday):
                if weekday == 1:
                    return [
                        {
                            "id": 1,
                            "nombre": "campo",
                            "dias": "1,2,3,4,5",
                            "hora_inicio": 8.0,
                            "hora_fin": 16.0,
                            "ocupacion": "campo",
                            "deporte": None,
                        }
                    ]
                return []

        capturado: dict = {}

        def _fake_predict(**kwargs):
            capturado.update(kwargs)
            return {
                "clase_final_label": "PELIGRO",
                "perfil": {"calor": {"prob_personalizada": 0.62}},
                "weather": {
                    "perfil_horario": [
                        {"hora": 8, "HI": 27.0, "temp": 28.0},
                        {"hora": 9, "HI": 28.0, "temp": 29.0},
                    ],
                    "current": {"t2m_c": 28.0, "rh": 60},
                },
                "modelos": {
                    "Formula": {"frio": {"wind_chill_c": 20}, "calor": {"heat_index_c": 28}}
                },
            }

        monkeypatch.setattr(mod, "_db", _DB())
        monkeypatch.setattr(mod, "predict_ensemble", _fake_predict)
        enviados = _stub_tg(monkeypatch)

        await mod._enviar_aviso_diario("1", 1)

        # La ocupación de la rutina entra en la predicción (no queda en ligera
        # genérica) y la ubicación sigue siendo la del perfil, no la de la rutina.
        perfil_pred = capturado["perfil"]
        assert perfil_pred["ocupacion"] == "campo"
        assert capturado["lat"] == 42.29 and capturado["lon"] == -8.81
        assert capturado["provincia"] == "Pontevedra"

        assert len(enviados) == 1
        assert "Campo 8:00-16:00 (Campo / agricultura x2.7)" in enviados[0]
        assert "PELIGRO" in enviados[0]
        assert "62%" in enviados[0]

    @pytest.mark.asyncio
    async def test_sin_rutinas_hoy_no_envia_nada(self, monkeypatch):
        import climasafeai.bot.telegram_bot as mod

        class _DB:
            def buscar_por_telegram(self, chat_id):
                return {"id": 7} if chat_id == "1" else None

            def obtener_perfil(self, _pid):
                return {"id": 7, "lat": 42.29, "lon": -8.81, "provincia": "Pontevedra"}

            def rutinas_por_dia(self, chat_id, weekday):
                return []  # hoy (miércoles) no toca ninguna

        def _no_deberia_calcular(**kwargs):
            raise AssertionError("no se calcula sin rutinas hoy")

        monkeypatch.setattr(mod, "_db", _DB())
        monkeypatch.setattr(mod, "predict_ensemble", _no_deberia_calcular)
        enviados = _stub_tg(monkeypatch)

        await mod._enviar_aviso_diario("1", 3)

        assert enviados == []

    def test_perfil_prediccion_desde_rutina_usa_la_ventana_de_la_rutina(self):
        from climasafeai.bot.telegram_bot import _perfil_prediccion_desde_rutina

        perfil = {
            "sexo": "hombre",
            "edad": 57,
            "aclimatado": False,
            "porcentaje_grasa": 20.5,
            "fototipo": 3,
            "comorbilidades": ["cardiovascular"],
            "farmacos": ["diureticos_asa"],
            "situacion_social": ["vive_solo"],
        }
        rutina = {"hora_inicio": 8.0, "hora_fin": 16.0, "deporte": "correr", "ocupacion": None}

        p = _perfil_prediccion_desde_rutina(perfil, rutina)

        assert p["hora_inicio"] == 8.0
        assert p["duracion_actividad_h"] == 8.0
        assert p["deporte"] == "correr"
        assert p["nivel_actividad"] == "muy_intensa"  # correr = 10.5 MET
        assert p["comorbilidades"] == {"cardiovascular"}
        assert p["farmacos"] == {"diureticos_asa"}

    def test_perfil_prediccion_desde_rutina_usa_ocupacion(self):
        from climasafeai.bot.telegram_bot import _perfil_prediccion_desde_rutina

        perfil = {
            "sexo": "hombre",
            "edad": 57,
            "aclimatado": False,
        }
        rutina = {"hora_inicio": 8.0, "hora_fin": 16.0, "deporte": None, "ocupacion": "campo"}

        p = _perfil_prediccion_desde_rutina(perfil, rutina)

        assert p["ocupacion"] == "campo"
        assert p.get("deporte") is None
