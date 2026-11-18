
import json
import sys
import traceback

print("[STANCE] Iniciando app.py", flush=True)

try:
    print("[STANCE] Importando módulo stance...", flush=True)

    from stance import ejecutar_job_stance, ejecutar_stance_por_target

    print("[STANCE] Módulo importado correctamente", flush=True)

    if __name__ == "__main__":
        print("[STANCE] Ejecutando trabajo...", flush=True)

        resultado = ejecutar_job_stance(
            "netvora_stance_posts",
            lambda: ejecutar_stance_por_target(
                max_comentarios=10,
                tamano_lote=10,
                max_publicaciones=3
            )
        )

        print(json.dumps(
            resultado,
            ensure_ascii=False,
            indent=2,
            default=str
        ), flush=True)

        sys.exit(1 if resultado.get("fallidos", 0) else 0)

except Exception:
    print("[STANCE ERROR] Falló la ejecución", flush=True)
    traceback.print_exc()
    sys.exit(1)
