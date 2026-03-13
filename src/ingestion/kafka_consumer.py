import json
import logging
from kafka import KafkaConsumer
from pydantic import ValidationError

from src.serving.schemas import LoanApplication

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IngestionConsumer")

class AlternativeDataConsumer:
    """
    Consumes asynchronous real-time behavioral and UPI events, validates them
    using Pydantic, and pushes viable data into the Feature Store.
    """
    def __init__(self, broker_url: str = "localhost:9092", topic: str = "raw-app-events"):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=[broker_url],
            auto_offset_reset='latest',
            enable_auto_commit=True,
            group_id='aurora-ingestion-group',
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        logger.info(f"Kafka consumer initialized on topic: {topic}")

    def run(self):
        """
        Blocking loop to continually consume messages.
        In K8s, this would run as an independent deployment worker.
        """
        for message in self.consumer:
            raw_payload = message.value
            try:
                # 1. Validation via Pydantic
                verified_app = LoanApplication(**raw_payload)
                
                # 2. Sink to Feature Store / Database
                self._sink_to_feature_store(verified_app)
                logger.info(f"Ingested validated payload for borrower: {verified_app.borrower_id}")
                
            except ValidationError as e:
                logger.error(f"Invalid Payload Received. Dropping. Error: {e.errors()}")
            except Exception as e:
                logger.error(f"Unexpected ingestion failure: {str(e)}")

    def _sink_to_feature_store(self, data: LoanApplication):
        """
        Hypothetical push to Redis or Feast.
        """
        # e.g., redis_client.hset(data.borrower_id, mapping=data.dict())
        pass

if __name__ == "__main__":
    # Mock entrypoint
    # consumer = AlternativeDataConsumer()
    # consumer.run()
    pass
