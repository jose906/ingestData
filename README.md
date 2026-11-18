# ingestData

## Configuración

Copia `.env.example` a `.env` para Docker Compose y completa los valores con credenciales vigentes. Para ejecutar Python directamente, exporta esas variables antes de iniciar el proceso.

- Para MySQL por TCP, configura `DB_HOST` y `DB_PORT`.
- Para Cloud SQL por socket, configura `INSTANCE_CONNECTION_NAME` y monta el socket del servicio bajo `/cloudsql`.
- Si quieres proteger las rutas de trabajo con un token, configura `INGEST_JOB_TOKEN` en el servicio y en el programador.

Con `docker compose -f docker_compose.yaml up --build`, Flask queda disponible en el puerto 8080. Llama a las rutas de trabajo con `GET`. Si configuraste `INGEST_JOB_TOKEN`, incluye el encabezado `Authorization: Bearer <INGEST_JOB_TOKEN>` en cada solicitud.

Para ejecutar ambos trabajos desde la terminal, usa `JOB_MODE=both python worker_replies.py`. También puedes elegir `ingest` o `replies`. Para Stance por publicación, ejecuta `python app.py`.

Instala `requirements-analysis.txt` para ejecutar localmente los scripts de embeddings, clustering y entrenamiento. Los modelos que usa Flask se cargan desde el directorio `model`.

Las credenciales presentes en versiones anteriores deben rotarse en sus servicios de origen. Los repositorios y las imágenes que se hayan publicado antes de este cambio pueden conservar copias.
