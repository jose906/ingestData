
import json
from stance import ejecutar_job_stance, ejecutar_stance_por_target


if __name__ == "__main__":
   a =  ejecutar_job_stance(
        "netvora_stance_posts",
        lambda: ejecutar_stance_por_target(
            max_comentarios=10,
            tamano_lote=10,
            max_publicaciones=3
        )
    )
   print(json.dumps(a, ensure_ascii=False, indent=2))