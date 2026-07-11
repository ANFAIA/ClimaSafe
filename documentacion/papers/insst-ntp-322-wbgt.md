# INSST — NTP 322: Valoración del riesgo de estrés térmico (índice WBGT)

> **Reutilización permitida con atribución** bajo el régimen español de
> reutilización de la información del sector público (Ley 37/2007 y RD
> 1495/2011): se permite reproducir con cita de la fuente, sin alterar el
> contenido y indicando la fecha de la última actualización. No es dominio
> público. Aquí se reproduce solo el **método** (hecho técnico) con la cita.
>
> Fuente: Instituto Nacional de Seguridad y Salud en el Trabajo (INSST).
> *NTP 322: Valoración del riesgo de estrés térmico: índice WBGT.* Serie NTP,
> 1994. https://www.insst.es/ (buscar "NTP 322").
>
> **Actualización:** la NTP 322 (1994) está **actualizada por la NTP 1189
> (2023)**, que incorpora la norma UNE-EN ISO 7243:2017. Usar la 1189 como
> referencia vigente; la 322 se conserva por continuidad histórica.

## Método (índice WBGT)

El índice WBGT discrimina de forma rápida si el riesgo de estrés térmico es
admisible o no. Se calcula combinando parámetros ambientales:

- **Exteriores con carga solar:**
  `WBGT = 0.7·THN + 0.2·TG + 0.1·TA`
- **Interiores o sin carga solar:**
  `WBGT = 0.7·THN + 0.3·TG`

donde `THN` = temperatura húmeda natural, `TG` = temperatura de globo,
`TA` = temperatura seca del aire.

El valor obtenido se compara con valores límite según el **consumo metabólico**
(carga de trabajo) y el grado de **aclimatación** — el mismo esquema que NIOSH
(ver `niosh-2016-106-heat.md`), aquí en su versión de normativa española.

## Nota sobre obligatoriedad

Las indicaciones de las NTP no son obligatorias salvo que estén recogidas en
una disposición normativa aplicable.

## Uso en el proyecto

Referencia de la **normativa española** aplicable al estrés térmico (base
científica del README) y del método WBGT, complementaria a NIOSH.
