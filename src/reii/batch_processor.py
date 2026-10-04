# -*- coding: utf-8 -*-
"""Procesamiento por lotes de transcripciones de entrevistas.

Pipeline de dos etapas sobre un grupo de archivos de transcripción:

1. **Ruta 1 — "AI Agent 1" (segmentación estructural)**: divide la
   transcripción en bloques secuenciales siguiendo la guía de 32 preguntas,
   conservando el texto literal.
2. **Ruta 2 — "AI Agent" (mapeo y extracción)**: extrae las respuestas y las
   mapea a las categorías del esquema de extracción.

Los resultados se persisten en una base JSON (por defecto
``data/entrevistas.json``) con la misma estructura que el resto del proyecto.
"""

from __future__ import annotations

import csv
import json
import os
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from reii.config import BATCH_DB_PATH, DATA_DIR, TRANSCRIPTS_DIR
from reii.guia_entrevista import guia_como_texto
from reii.ia_discursiva import call_deepseek_structured

# ---------------------------------------------------------------------------
# Esquemas JSON (Ruta 1 y Ruta 2)
# ---------------------------------------------------------------------------

SCHEMA_SEGMENTACION: Dict = {
    "type": "object",
    "properties": {
        "segmentos": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "numero_pregunta": {"type": "integer"},
                    "seccion_entrevista": {
                        "type": "string",
                        "enum": [
                            "I. Datos sociodemográficos y perfil profesional",
                            "II. Concepciones e información general sobre el cambio climático",
                            "III. Percepciones del entorno y efectos directos",
                            "IV. El rol de la educación universitaria",
                            "V. Acciones de mitigación y proyección docente",
                            "VI. Perspectiva futura y responsabilidad social",
                            "TEXTO_NO_ASIGNADO",
                        ],
                    },
                    "texto_literal": {"type": "string"},
                },
                "required": ["numero_pregunta", "seccion_entrevista", "texto_literal"],
            },
        }
    },
    "required": ["segmentos"],
}

SCHEMA_EXTRACCION: Dict = {
    "type": "object",
    "properties": {
        "datos_sociodemograficos": {
            "type": "object",
            "properties": {
                "Alias": {"type": "string"},
                "sexo": {"type": "string"},
                "edad": {"type": "string"},
                "dependientes_economicos": {"type": "string"},
                "ocupacion": {"type": "string"},
                "lugar_procedencia": {"type": "string"},
                "distrito_residencia": {"type": "string"},
                "carrera_especialidad": {"type": "string"},
                "formacion_adicional": {"type": "string"},
                "motivo_eleccion_carrera": {"type": "string"},
                "proyeccion_5_anos": {"type": "string"},
            },
            "required": [
                "Alias",
                "sexo",
                "edad",
                "dependientes_economicos",
                "ocupacion",
                "lugar_procedencia",
                "distrito_residencia",
                "carrera_especialidad",
                "formacion_adicional",
                "motivo_eleccion_carrera",
                "proyeccion_5_anos",
            ],
        },
        "concepciones_cambio_climatico": {
            "type": "object",
            "properties": {
                "pensamiento_inicial": {"type": "string"},
                "causas_mencionadas": {"type": "array", "items": {"type": "string"}},
                "medios_informacion_recibida": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "medios_mas_importantes": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "materiales_consultados": {"type": "string"},
            },
            "required": [
                "pensamiento_inicial",
                "causas_mencionadas",
                "medios_informacion_recibida",
                "medios_mas_importantes",
                "materiales_consultados",
            ],
        },
        "percepciones_entorno": {
            "type": "object",
            "properties": {
                "diferencias_clima_percibidas": {"type": "string"},
                "tiempo_estimado_cambios": {"type": "string"},
                "significado_y_causas_atribuidas": {"type": "string"},
                "influencia_vida_personas": {"type": "string"},
                "grupos_vulnerables": {"type": "array", "items": {"type": "string"}},
                "afectacion_salud": {"type": "string"},
                "experiencia_personal_familiar": {"type": "string"},
            },
            "required": [
                "diferencias_clima_percibidas",
                "tiempo_estimado_cambios",
                "significado_y_causas_atribuidas",
                "influencia_vida_personas",
                "grupos_vulnerables",
                "afectacion_salud",
                "experiencia_personal_familiar",
            ],
        },
        "rol_educacion_universitaria": {
            "type": "object",
            "properties": {
                "conocimientos_suficientes_universidad": {"type": "string"},
                "temas_desarrollados_asignaturas": {"type": "string"},
                "importancia_abordar_tema": {"type": "string"},
                "cambios_propuestos_plan_estudios": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "conocimientos_suficientes_universidad",
                "temas_desarrollados_asignaturas",
                "importancia_abordar_tema",
                "cambios_propuestos_plan_estudios",
            ],
        },
        "acciones_mitigacion": {
            "type": "object",
            "properties": {
                "acciones_conocidas_entorno": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "acciones_sostenibles_personales": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "interes_ensenanza_futuro_docente": {"type": "string"},
            },
            "required": [
                "acciones_conocidas_entorno",
                "acciones_sostenibles_personales",
                "interes_ensenanza_futuro_docente",
            ],
        },
        "perspectiva_futura": {
            "type": "object",
            "properties": {
                "proyeccion_futura_cambios": {"type": "string"},
                "percepcion_accion_pais_region": {"type": "string"},
                "acciones_necesarias_sociedad": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": [
                "proyeccion_futura_cambios",
                "percepcion_accion_pais_region",
                "acciones_necesarias_sociedad",
            ],
        },
    },
    "required": [
        "datos_sociodemograficos",
        "concepciones_cambio_climatico",
        "percepciones_entorno",
        "rol_educacion_universitaria",
        "acciones_mitigacion",
        "perspectiva_futura",
    ],
}

# ---------------------------------------------------------------------------
# Prompts de sistema (Ruta 1 y Ruta 2)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_SEGMENTACION: str = """\
Eres un asistente de investigación cualitativa especializado en segmentación estructural de entrevistas.

Tu tarea es segmentar la transcripción de una entrevista en bloques secuenciales siguiendo la guía de 32 preguntas que se te proporciona.

Reglas estrictas:
- Extrae el texto LITERAL de la entrevista. No resumas, no parafrasees, no clasifiques y no omitas contenido.
- Corrige únicamente errores ortográficos y de puntuación evidentes del software de transcripción, pero NO cambies las palabras ni el tono del entrevistado.
- Si una palabra fue mal transcrita de forma evidente, sustitúyela por la opción contextualmente correcta.
- Si el entrevistador interrumpe con un comentario menor en medio de una respuesta, coloca el texto del entrevistador entre paréntesis ( ) dentro del mismo bloque. No uses etiquetas "Entrevistado:" ni "Entrevistador:".
- Si una pregunta de la guía no fue formulada o está ausente, IGNÓRALA: no generes un segmento vacío.
- El texto huérfano al inicio o al final que no corresponda a ninguna pregunta debe asignarse con numero_pregunta: 0 y seccion_entrevista: "TEXTO_NO_ASIGNADO".
- Devuelve ÚNICAMENTE un objeto JSON válido que cumpla el esquema. Cero texto plano antes o después del JSON."""

SYSTEM_PROMPT_EXTRACCION: str = """\
Eres un asistente de investigación cualitativa especializado en el análisis de entrevistas semiestructuradas.

Tu tarea es extraer y mapear las respuestas de la entrevista a las categorías del esquema de salida, usando la guía de 32 preguntas como referencia.

Reglas estrictas:
- Limpia los errores de transcripción sobre la marcha (ortografía, palabras mal transcritas según el contexto, puntuación) sin alterar el significado.
- Mapea de forma inteligente las respuestas a las categorías, sin importar en qué parte del texto aparezcan.
- CERO alucinaciones: si una pregunta de la guía no fue respondida o no se aborda en la entrevista, escribe exactamente "NO ESTÁ PRESENTE" para los campos de texto y [] para los campos de lista.
- Devuelve ÚNICAMENTE un objeto JSON válido según el esquema. Sin saludos, explicaciones ni etiquetas."""

# ---------------------------------------------------------------------------
# Constructores de mensajes
# ---------------------------------------------------------------------------


def build_prompt_segmentacion(transcripcion: str) -> List[Dict[str, str]]:
    """Construye los mensajes (system + user) para la Ruta 1 (segmentación)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SEGMENTACION},
        {
            "role": "user",
            "content": (
                f"Guía de entrevista:\n{guia_como_texto()}\n\n"
                f"Entrevista actual:\n{transcripcion}\n\n"
                "Devuelve únicamente el objeto JSON válido según el esquema."
            ),
        },
    ]


def build_prompt_extraccion(transcripcion: str) -> List[Dict[str, str]]:
    """Construye los mensajes (system + user) para la Ruta 2 (extracción)."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT_EXTRACCION},
        {
            "role": "user",
            "content": (
                f"Guía de entrevista:\n{guia_como_texto()}\n\n"
                f"Entrevista actual:\n{transcripcion}\n\n"
                "Devuelve únicamente el objeto JSON válido según el esquema."
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Procesador por lotes
# ---------------------------------------------------------------------------

# Centinela de fallo devuelto por call_deepseek_structured cuando agota
# reintentos o no puede validar el esquema.
_FALLO_SENTINEL: Dict = {"annotations": []}


def _llave_maestra(nombre: str) -> str:
    """Limpia extensiones igual que ``obtener_llave_maestra`` del workflow.

    ``"entrevista_01.txt.json"`` / ``"entrevista_01.txt"`` → ``"entrevista_01"``.
    """
    return str(nombre).replace(".txt.json", "").replace(".json", "").replace(".txt", "")


def colapsar_repeticiones(texto: str) -> str:
    """Colapsa artefactos de bucle del software de transcripción.

    Algunas transcripciones quedan atascadas repitiendo la misma palabra o
    frase (p. ej. ``"no, no, no, no, no..."`` o ``"¿Qué es eso? ¿Qué es eso?..."``).
    Ese ruido confunde al modelo y lo lleva a devolver ``{}``. Esta limpieza
    solo afecta al texto que se envía al modelo; el original se conserva.
    """
    # 1) Palabra repetida con comas: "no, no, no, no, no" → "no"
    texto = re.sub(r"\b(\w+)(?:,\s*\1\b){4,}", r"\1", texto)
    # 2) Frase repetida en una misma línea (3+ veces la misma oración)
    lineas = texto.splitlines()
    nuevas = []
    for linea in lineas:
        oraciones = re.findall(r"[^.!?]+[.!?]", linea)
        if len(oraciones) >= 3 and len(set(o.strip() for o in oraciones)) == 1:
            linea = oraciones[0]
        nuevas.append(linea)
    return "\n".join(nuevas)


class BatchProcessor:
    """Procesa un grupo de transcripciones con el pipeline de dos etapas.

    Parámetros
    ----------
    db_path:
        Ruta del archivo JSON donde se persisten los registros.
    max_retries:
        Reintentos por llamada al modelo (se pasa a
        :func:`call_deepseek_structured`).
    temperature:
        Temperatura de muestreo del modelo.
    """

    def __init__(
        self,
        db_path: str = BATCH_DB_PATH,
        max_retries: int = 3,
        temperature: float = 0.2,
    ) -> None:
        self.db_path: str = db_path
        self.max_retries: int = max_retries
        self.temperature: float = temperature
        self._ensure_db()

    # ── Persistencia ──────────────────────────────────────────────────
    @staticmethod
    def _write_json(path: Path, data: Dict) -> None:
        """Escribe JSON de forma atómica (temp + replace).

        Evita que una lectura concurrente (p. ej. la UI mientras el worker
        guarda) vea un archivo a medio escribir.
        """
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    def _ensure_db(self) -> None:
        """Crea el archivo JSON de la base si no existe (``{"entrevistas": []}``)."""
        path = Path(self.db_path)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            self._write_json(path, {"entrevistas": []})

    def load_all(self) -> List[Dict]:
        """Devuelve todos los registros persistidos.

        Reintenta brevemente si el archivo está siendo reescrito por otro
        hilo (el worker de la UI guarda mientras la página lee).
        """
        self._ensure_db()
        for _ in range(5):
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("entrevistas", [])
            except (json.JSONDecodeError, OSError):
                time.sleep(0.1)
        return []

    def save_record(self, record: Dict) -> None:
        """Inserta o actualiza (upsert por ``id``) un registro en la base."""
        records = self.load_all()
        idx = next(
            (i for i, r in enumerate(records) if r.get("id") == record.get("id")),
            None,
        )
        if idx is None:
            records.append(record)
        else:
            records[idx] = record
        self._write_json(Path(self.db_path), {"entrevistas": records})

    def delete_record(self, record_id: str) -> None:
        """Elimina un registro por su ``id``."""
        records = self.load_all()
        records = [r for r in records if r.get("id") != record_id]
        self._write_json(Path(self.db_path), {"entrevistas": records})

    # ── Descubrimiento y lectura de archivos ─────────────────────────
    def discover_files(self, directory: str = None) -> List[Path]:
        """Escanea el directorio de transcripciones en busca de archivos.

        Acepta ``.txt``, ``.json`` y ``.md``. Devuelve la lista ordenada.
        """
        base = Path(directory) if directory else Path(TRANSCRIPTS_DIR)
        if not base.exists():
            return []
        db_path = Path(self.db_path).resolve()
        db_name = Path(self.db_path).name
        return sorted(
            p
            for p in base.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".txt", ".json", ".md"}
            and p.resolve() != db_path
            and p.name != db_name
        )

    def read_transcript(self, path: Path) -> str:
        """Lee el contenido de una transcripción.

        - ``.txt`` / ``.md``: texto plano.
        - ``.json``: intenta las claves ``texto`` / ``transcripcion`` /
          ``content`` o un valor string en la raíz.
        """
        path = Path(path)
        if path.suffix.lower() == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, str):
                return data
            if isinstance(data, dict):
                for key in ("texto", "transcripcion", "content"):
                    if key in data and isinstance(data[key], str):
                        return data[key]
            return json.dumps(data, ensure_ascii=False, indent=2)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ── Pipeline de dos etapas ───────────────────────────────────────
    def process_file(
        self, path: Path, log_cb: Optional[Callable[[str], None]] = None
    ) -> Dict:
        """Procesa una transcripción con las dos rutas y arma el registro.

        Si alguna ruta devuelve el centinela de fallo (``{"annotations": []}``),
        el registro se marca con ``estado: "error"`` y se guarda el resultado
        crudo devuelto por el modelo.

        ``log_cb`` (opcional) recibe cada mensaje de progreso/error del modelo;
        los mensajes también se acumulan en el campo ``log`` del registro.
        """
        path = Path(path)
        log: List[str] = []

        def _cb(msg: str) -> None:
            log.append(msg)
            if log_cb is not None:
                log_cb(msg)

        try:
            transcripcion = self.read_transcript(path)
            # Limpia artefactos de bucle del STT antes de enviar al modelo.
            # El texto original se conserva intacto en el registro.
            transcripcion = colapsar_repeticiones(transcripcion)

            segmentos = call_deepseek_structured(
                build_prompt_segmentacion(transcripcion),
                SCHEMA_SEGMENTACION,
                max_retries=self.max_retries,
                temperature=self.temperature,
                log_cb=_cb,
            )
            extraccion = call_deepseek_structured(
                build_prompt_extraccion(transcripcion),
                SCHEMA_EXTRACCION,
                max_retries=self.max_retries,
                temperature=self.temperature,
                log_cb=_cb,
            )

            estado = "completado"
            if segmentos == _FALLO_SENTINEL or extraccion == _FALLO_SENTINEL:
                estado = "error"

            return {
                "id": str(uuid.uuid4()),
                "archivo": path.name,
                "ruta": str(path),
                "fecha_procesamiento": datetime.now().isoformat(timespec="seconds"),
                "estado": estado,
                "segmentos": segmentos,
                "extraccion": extraccion,
                "log": log,
            }
        except Exception as e:
            # Un archivo ilegible o un fallo inesperado no debe abortar el lote.
            log.append(f"❌ {type(e).__name__}: {e}")
            return {
                "id": str(uuid.uuid4()),
                "archivo": path.name,
                "ruta": str(path),
                "fecha_procesamiento": datetime.now().isoformat(timespec="seconds"),
                "estado": "error",
                "error": f"{type(e).__name__}: {e}",
                "segmentos": {},
                "extraccion": {},
                "log": log,
            }

    def process_batch(
        self,
        files: List[Path],
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> List[Dict]:
        """Procesa una lista de archivos y persiste cada registro.

        ``progress_cb(i, n, name)`` se invoca antes de procesar cada archivo
        (``i`` es 1-based, ``n`` el total, ``name`` el nombre del archivo).
        ``log_cb`` se reenvía a :meth:`process_file` para cada archivo.
        """
        records: List[Dict] = []
        total = len(files)
        for i, path in enumerate(files, start=1):
            if progress_cb is not None:
                progress_cb(i, total, path.name)
            record = self.process_file(path, log_cb=log_cb)
            self.save_record(record)
            records.append(record)
        return records

    # ── Exportación al workflow principal ───────────────────────────
    def exportar_para_workflow(self, output_dir: str = None) -> Dict[str, List[Path]]:
        """Exporta los registros completados al formato que consume el workflow.

        Por cada entrevista con ``estado == "completado"`` escribe:

        - ``<output_dir>/tmp/<llave>.txt.json`` con la estructura
          ``{"output": {"segmentos": [...]}}`` que lee el workflow.
        - ``<output_dir>/<llave>.txt`` con la transcripción cruda.
        - ``<output_dir>/Refined_Database.csv`` con la metadata
          sociodemográfica extraída (Ruta 2), separada por ``;``.

        Devuelve un dict con las listas de archivos escritos
        (``json``, ``txt``, ``csv``).
        """
        base = Path(output_dir) if output_dir else Path(DATA_DIR) / "txt_outputs"
        tmp_dir = base / "tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        json_paths: List[Path] = []
        txt_paths: List[Path] = []
        filas_meta: List[Dict[str, str]] = []

        for rec in self.load_all():
            if rec.get("estado") != "completado":
                continue
            segmentos = rec.get("segmentos", {}).get("segmentos", [])
            if not segmentos:
                continue

            llave = _llave_maestra(rec.get("archivo", ""))

            # 1) JSON con la estructura que lee el workflow
            json_path = tmp_dir / f"{llave}.txt.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(
                    {"output": {"segmentos": segmentos}},
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            json_paths.append(json_path)

            # 2) Transcripción cruda (el workflow la lee junto al JSON)
            txt_path = base / f"{llave}.txt"
            if not txt_path.exists():
                origen = Path(rec.get("ruta", ""))
                if not origen.exists():
                    origen = Path(TRANSCRIPTS_DIR) / rec.get("archivo", "")
                if origen.exists():
                    txt_path.write_text(self.read_transcript(origen), encoding="utf-8")
            txt_paths.append(txt_path)

            # 3) Fila de metadata para Refined_Database.csv
            filas_meta.append(self._fila_metadata(llave, rec))

        csv_path = base / "Refined_Database.csv"
        if filas_meta:
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "Documento Fuente",
                        "Edad_Cat",
                        "Sexo",
                        "Dependientes_Cat",
                        "Ocupacion_Cat",
                        "Procedencia_Cat",
                    ],
                    delimiter=";",
                )
                writer.writeheader()
                writer.writerows(filas_meta)

        return {
            "json": json_paths,
            "txt": txt_paths,
            "csv": [csv_path] if filas_meta else [],
        }

    def _fila_metadata(self, llave: str, rec: Dict) -> Dict[str, str]:
        """Mapea la extracción (Ruta 2) a una fila de ``Refined_Database.csv``."""
        socio = rec.get("extraccion", {}).get("datos_sociodemograficos", {}) or {}

        def _v(campo: str) -> str:
            valor = socio.get(campo, "")
            return "" if valor in (None, "NO ESTÁ PRESENTE") else str(valor)

        return {
            "Documento Fuente": llave,
            "Edad_Cat": _v("edad"),
            "Sexo": _v("sexo"),
            "Dependientes_Cat": _v("dependientes_economicos"),
            "Ocupacion_Cat": _v("ocupacion"),
            "Procedencia_Cat": _v("lugar_procedencia"),
        }
