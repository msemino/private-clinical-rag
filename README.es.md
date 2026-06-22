# private-clinical-rag

**Un pipeline RAG 100% local y con trazabilidad por citas para manuales de referencia críticos — corre en hardware propio, con cero fuga de datos.**

[English](README.md) · [Español](README.es.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![ChromaDB](https://img.shields.io/badge/Vector%20store-ChromaDB-FF6F61)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-000000?logo=ollama&logoColor=white)
![Privacy](https://img.shields.io/badge/Fuga%20de%20datos-cero-3DA639)
![License](https://img.shields.io/badge/License-MIT-blue)

---

## Por qué existe

La mayoría de los demos de "RAG en 50 líneas" hacen tres cosas que los descartan
para dominios serios: cargan los documentos como si fueran prosa, confían en la
similitud vectorial cruda, y responden incluso cuando no deberían. En un entorno
de referencia crítico —clínico, legal, regulatorio— si una respuesta no tiene
trazabilidad, no es una funcionalidad: es deuda técnica.

Este proyecto es lo opuesto. Trata al manual como una **jerarquía de unidades
lógicas**, enriquece cada fragmento con metadatos, re-rankea más allá del coseno,
**cita sus fuentes** y **se niega a responder** cuando el contexto recuperado no
respalda una respuesta. Todo —embeddings, vector store y el LLM— corre **local**.
Ningún documento, consulta ni embedding sale jamás de la máquina.

> ### El caso que lo originó: "Bianca"
> Este pipeline nació de **Bianca**, un asistente privado sobre el manual de
> salud mental **DSM-5**. Como el DSM-5 tiene copyright, **este repositorio no
> incluye contenido del DSM-5** —ni el corpus, ni excerpts—. Incluye la
> *arquitectura* y un pequeño documento de ejemplo **sintético** para correrlo en
> 30 segundos; después lo apuntás a tu propio corpus con licencia. Ver
> [Traé tu propio corpus](#traé-tu-propio-corpus).

---

## Arquitectura

```mermaid
flowchart LR
    subgraph Local["🖥️  Tu máquina — nada sale de acá"]
        DOC[/"Tu manual<br/>(Markdown)"/] --> ING["Ingesta<br/>chunking jerárquico<br/>+ metadatos"]
        ING -->|embeddings| EMB{{"Embedder<br/>Ollama · local"}}
        EMB --> VDB[("ChromaDB<br/>persistente, en disco")]
        Q(["Pregunta"]) --> RET["Retrieve<br/>vector search → re-rank"]
        VDB --> RET
        RET --> GATE{"score top<br/>≥ MIN_SCORE?"}
        GATE -- no --> REF["Rechaza:<br/>'no está en el manual'"]
        GATE -- sí --> LLM{{"LLM<br/>Ollama · local"}}
        LLM --> ANS["Respuesta<br/>+ citas [S#]"]
    end
```

---

## Arranque rápido

### 1. Ejecutalo offline en 30 segundos (sin GPU, sin Ollama)

```bash
python -m venv .venv && . .venv/Scripts/activate   # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python -m src.ask selftest
```

El self-test indexa el sample sintético con un embedder offline determinístico y
verifica todo el contrato: chunks indexados → sección correcta recuperada →
respuesta que cita su fuente → pregunta off-topic **rechazada**.

### 2. Ejecutalo de verdad (Ollama local)

```bash
ollama pull nomic-embed-text
ollama pull llama3.1:8b

cp .env.example .env
python -m src.ask ingest data/sample/clinical_handbook_sample.md
python -m src.ask ask "¿Cómo se evalúa un episodio de pánico?"
```

Cada respuesta vuelve con el path de sección de cada fuente usada.

---

## Cómo funciona (lo que importa)

| Decisión | Por qué |
|---|---|
| **Chunking jerárquico** | Los documentos se cortan siguiendo el árbol de títulos; cada fragmento conserva su path de sección completo. Un fragmento que perdería el contexto del padre se **descarta**, no se indexa a ciegas. |
| **"DNI" de metadatos por chunk** | Fuente, título, path de sección, ordinal — para aplicar filtros `where` quirúrgicos *antes* de que el LLM vea la consulta. |
| **Re-ranking, no similitud cruda** | Los candidatos del vector store se re-rankean con una mezcla transparente de coseno, solapamiento léxico y boost por match de título. Auditable, microsegundos, sin dependencia de cross-encoder. |
| **Compuerta de rechazo** | Si el mejor chunk puntúa por debajo de `MIN_SCORE`, devuelve *"no lo encuentro en el manual"* en vez de alucinar. |
| **Citas** | El prompt prohíbe conocimiento externo y exige tags `[S#]`; las fuentes se imprimen con cada respuesta. |
| **Local por construcción** | Embeddings (Ollama), vector store (ChromaDB en disco) y generación (Ollama) corren en tu hardware. **Cero fuga de datos.** |
| **Embedder enchufable** | `OllamaEmbedder` para producción, `HashEmbedder` para offline/CI — misma interfaz, pipeline testeable sin ningún servicio corriendo. |

---

## Traé tu propio corpus

El repo incluye un sample **sintético** (`data/sample/clinical_handbook_sample.md`)
—contenido ficticio escrito para este proyecto, **no** el DSM-5 ni ningún manual
con copyright—. Para usarlo de verdad:

1. Poné tus documentos Markdown **con licencia** bajo `data/` (la carpeta
   `data/private/` está git-ignored por default).
2. `python -m src.ask ingest "data/private/*.md"`
3. `python -m src.ask ask "tu pregunta"`

> ⚠️ **Sos responsable de los derechos del corpus que indexes.** No subas material
> con copyright a un repo público. Esta herramienta mantiene tu corpus local
> justamente para que siga siendo privado.

> 🩺 **No es consejo médico.** Es una herramienta de recuperación y citado sobre
> un manual, no un sistema de diagnóstico.

---

## Licencia

[MIT](LICENSE) para el código. El sample sintético es contenido original bajo la
misma licencia. Cualquier corpus que *vos* agregues se rige por *su* licencia.
