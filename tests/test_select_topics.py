import os
import unittest
from unittest.mock import Mock, patch

import selectTopics


class SelectTopicsTest(unittest.TestCase):
    def test_select_topics_keeps_ambiguous_tweets_pending_without_loading_references(self):
        import main

        tweets = [
            {"tweetid": 1, "text": "El municipio anunció el plan vial", "embedding": [1, 0]},
            {"tweetid": 2, "text": "Presentaron las nuevas medidas", "embedding": [0.95, 0.31]},
            {"tweetid": 3, "text": "https://t.co/enlace", "embedding": None},
        ]
        topics = [
            {"topic_id": 10, "topic_name": "Nuevo plan vial", "active": 1, "embedding": [1, 0]},
            {"topic_id": 20, "topic_name": "Presupuesto municipal", "active": 1, "embedding": [0.8, 0.6]},
        ]
        db = Mock()
        db.cursor.return_value.fetchone.return_value = (1,)

        with (
            patch.dict(os.environ, {"INGEST_JOB_TOKEN": ""}),
            patch.object(main, "get_db_connection", return_value=db),
            patch.object(selectTopics, "get_tweet_embedding", return_value=tweets),
            patch.object(selectTopics, "get_topic_embeddings", return_value=topics),
            patch.object(selectTopics, "get_especial_topics", return_value={"__SOLO_LINK__": 90}),
            patch.object(selectTopics, "insert_tweets_topic", return_value={1, 3}) as guardar,
            patch.object(
                selectTopics.helper, "cargar_referencias_topics",
                side_effect=AssertionError("La ruta intentó leer referencias históricas"),
            ) as cargar_referencias,
        ):
            response = main.app.test_client().get("/selectTopics")

        self.assertEqual(response.status_code, 200, response.json)
        self.assertEqual(response.json["motivos_no_asignados"], {"ambiguous": 1})
        self.assertEqual(response.json["tweets_asignados"], 2)
        self.assertEqual(response.json["tweets_asignados_a_topicos"], 1)
        cargar_referencias.assert_not_called()
        guardar.assert_called_once()
        asignados = guardar.call_args.args[0]
        self.assertCountEqual(zip(asignados["tweetid"], asignados["topic_id"]), [(1, 10), (3, 90)])
