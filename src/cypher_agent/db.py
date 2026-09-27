import atexit
from functools import lru_cache

from neo4j import Driver, GraphDatabase

from cypher_agent.config import settings


@lru_cache(maxsize=1)
def get_driver() -> Driver:
    driver = GraphDatabase.driver(
        settings.neo4j_uri, auth=(settings.neo4j_username, settings.neo4j_password)
    )
    driver.verify_connectivity()
    atexit.register(driver.close)
    return driver
