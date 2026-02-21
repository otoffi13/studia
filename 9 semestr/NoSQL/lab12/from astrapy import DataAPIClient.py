from astrapy.info import (
    CollectionDefinition,
    CollectionVectorOptions,
    VectorServiceOptions,
)
from astrapy.constants import VectorMetric

collection = database.create_collection(
    "test_collection",
    definition=CollectionDefinition(
        vector=CollectionVectorOptions(
            dimension=1536,
            metric=VectorMetric.DOT_PRODUCT,
            service=VectorServiceOptions(
                provider="openai",
                model_name="text-embedding-3-small",
                authentication={
                    "providerKey": "API_KEY_NAME",
                },
            ),
        ),
    ),
)