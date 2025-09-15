import uuid
from weaviate.classes.query import Filter, QueryReference

from elysia.objects import Retrieval

# Client Manager
from elysia.util.client import ClientManager

# Utilities
from elysia.util.parsing import format_dict_to_serialisable


class MessageRetrieval(Retrieval):
    def __init__(
        self,
        objects: list[dict],
        metadata: dict,
        mapping: dict | None = None,
        **kwargs,
    ) -> None:
        for obj in objects:
            obj["relevant"] = False

        Retrieval.__init__(
            self,
            objects,
            payload_type="message",
            metadata=metadata,
            mapping=mapping,
            unmapped_keys=[
                "uuid",
                "ELYSIA_SUMMARY",
                "collection_name",
                "relevant",
                "_REF_ID",
            ],
            **kwargs,
        )


class ConversationRetrieval(Retrieval):
    def __init__(
        self,
        objects: list[dict],
        metadata: dict,
        mapping: dict | None = None,
        **kwargs,
    ) -> None:
        Retrieval.__init__(
            self,
            objects,
            payload_type="conversation",
            metadata=metadata,
            mapping=mapping,
            unmapped_keys=[
                "uuid",
                "ELYSIA_SUMMARY",
                "collection_name",
                "relevant",
                "_REF_ID",
            ],
            **kwargs,
        )
        self.async_init_completed = False

    async def async_init(self, client_manager: ClientManager) -> None:
        if not self.async_init_completed:
            await self._return_all_messages_in_conversation(client_manager)

    async def _fetch_items_in_conversation(
        self,
        conversation_id: str,
        message_id: int,
        conversation_id_field_name: str,
        message_id_field_name: str,
        metadata: dict,
        client_manager: ClientManager,
    ) -> list[dict]:
        """
        Use Weaviate to fetch all messages in a conversation based on the conversation ID.
        """
        assert (
            "collection_name" in metadata
        ), "collection_name is required for fetching other messages in a conversation"

        async with client_manager.connect_to_async_client() as client:
            collection = client.collections.get(metadata["collection_name"])
            items_in_conversation = await collection.query.fetch_objects(
                filters=Filter.by_property(conversation_id_field_name).equal(
                    conversation_id
                )
            )

        output = []
        for obj in items_in_conversation.objects:
            output.append(obj.properties)
            output[-1]["uuid"] = str(obj.uuid)
            if output[-1][message_id_field_name] == message_id:
                output[-1]["relevant"] = True
            else:
                output[-1]["relevant"] = False

        return output

    async def _return_all_messages_in_conversation(
        self, client_manager: ClientManager
    ) -> list[dict]:
        """
        Return all messages in a conversation based on the response from Weaviate.
        """

        returned_objects = []
        conversations_seen = set()
        for o in self.objects:

            if o[self.mapping["conversation_id"]] in conversations_seen:
                continue

            conversations_seen.add(o[self.mapping["conversation_id"]])

            items_in_conversation = await self._fetch_items_in_conversation(
                o[self.mapping["conversation_id"]],
                o[self.mapping["message_id"]],
                self.mapping["conversation_id"],
                self.mapping["message_id"],
                self.metadata,
                client_manager,
            )
            # Check if all message_id values can be converted to int for sorting
            can_sort_as_int = True
            for x in items_in_conversation:
                try:
                    int(x[self.mapping["message_id"]])
                except (ValueError, TypeError, KeyError):
                    can_sort_as_int = False
                    break

            if can_sort_as_int:
                items_in_conversation.sort(
                    key=lambda x: int(x[self.mapping["message_id"]])
                )
            else:
                items_in_conversation.sort(key=lambda x: x[self.mapping["message_id"]])

            returned_objects.append(
                {
                    "messages": items_in_conversation,
                    "conversation_id": o[self.mapping["conversation_id"]],
                }
            )

        self.objects = returned_objects
        self.async_init_completed = True

        return returned_objects

    def to_json(self, mapping: bool = False) -> list[dict]:
        assert (
            self.async_init_completed
        ), "ConversationRetrieval not initialized, need to run .async_init(client_manager)"

        if self.mapping is not None and mapping:
            output_objects = []
            for obj in self.objects:  # outer level, conversation
                output_objects.append(
                    {
                        "conversation_id": obj["conversation_id"],
                        "messages": [],
                    }
                )
                for message in obj["messages"]:  # inner level, message
                    output_objects[-1]["messages"].append(
                        {
                            key: message[self.mapping[key]]
                            for key in self.mapping
                            if self.mapping[key] != ""
                        }
                    )
                    output_objects[-1]["messages"][-1].update(
                        {
                            key: message[key]
                            for key in self.unmapped_keys
                            if key in message
                        }
                    )
        else:
            output_objects = self.objects

        for object in output_objects:
            for message in object["messages"]:
                format_dict_to_serialisable(message)

        return output_objects


class DocumentRetrieval(Retrieval):
    def __init__(
        self,
        objects: list[dict],
        metadata: dict,
        mapping: dict | None = None,
        **kwargs,
    ) -> None:
        assert (
            "collection_name" in metadata
        ), "collection_name is required for DocumentRetrieval"

        if "chunked" not in metadata:
            metadata["chunked"] = False

        for obj in objects:
            obj["collection_name"] = metadata["collection_name"]
            if "chunk_spans" not in obj:
                obj["chunk_spans"] = []

        Retrieval.__init__(
            self,
            objects,
            payload_type="document",
            metadata=metadata,
            mapping=mapping,
            unmapped_keys=[
                "uuid",
                "ELYSIA_SUMMARY",
                "collection_name",
                "chunk_spans",
                "chunk_uuid",
                "_REF_ID",
            ],
            **kwargs,
        )
        self.async_init_completed = False

    async def async_init(self, client_manager: ClientManager) -> None:
        if not self.async_init_completed:
            await self._get_related_documents(client_manager)

    async def _get_related_documents(self, client_manager: ClientManager) -> None:
        """
        Get the related full documents for the chunked documents in the DocumentRetrieval object.
        """
        if self.metadata["chunked"]:
            chunked_collection_name = (
                f"ELYSIA_CHUNKED_{self.metadata['collection_name'].lower()}__"
            )
            async with client_manager.connect_to_async_client() as client:
                if await client.collections.exists(chunked_collection_name):
                    chunked_collection = client.collections.get(chunked_collection_name)

                    # re-retrieve the exact same items but with the full document references
                    chunked_response = (
                        await chunked_collection.query.fetch_objects_by_ids(
                            [object["uuid"] for object in self.objects],
                            return_references=QueryReference(link_on="fullDocument"),
                        )
                    )

                    # Loop over all references in this new response and add them to the full_docs dictionary
                    full_docs = {}
                    for object in chunked_response.objects:
                        # retrieve cross references from this current chunked response object
                        if "fullDocument" in object.references:
                            references = object.references["fullDocument"].objects

                            # this list is length one (only one full doc per chunk)
                            for full_document in references:

                                # each chunk is attached to a full doc, but can have multiple chunks per full doc
                                if str(full_document.uuid) not in full_docs:
                                    full_docs[str(full_document.uuid)] = {
                                        **full_document.properties,
                                        "uuid": str(full_document.uuid),
                                        "collection_name": self.metadata[
                                            "collection_name"
                                        ],
                                        "chunk_spans": [],
                                    }

                                chunk_spans: list[int] = object.properties["chunk_spans"]  # type: ignore

                                # chunk comes from `object` which is the chunk (outer level loop)
                                full_docs[str(full_document.uuid)][
                                    "chunk_spans"
                                ].append(
                                    {
                                        "start": chunk_spans[0],
                                        "end": chunk_spans[1],
                                        "uuid": str(object.uuid),
                                    }
                                )

                    self.full_documents = list(full_docs.values())
                else:
                    self.full_documents = self.objects
        else:
            self.full_documents = self.objects
        self.async_init_completed = True

    def full_documents_to_json(self, mapping: bool = False) -> list[dict]:
        assert all(
            isinstance(obj, dict) for obj in self.full_documents
        ), "All objects must be dictionaries"

        if mapping and self.mapping is not None:
            output_objects = []
            for obj in self.full_documents:
                output_objects.append(
                    {
                        key: obj[self.mapping[key]]
                        for key in self.mapping
                        if self.mapping[key] != ""
                    }
                )
                output_objects[-1].update(
                    {key: obj[key] for key in self.unmapped_keys if key in obj}
                )
        else:
            output_objects = self.full_documents

        for object in output_objects:
            format_dict_to_serialisable(object)

        return output_objects

    async def to_frontend(
        self, user_id: str, conversation_id: str, query_id: str
    ) -> dict:
        """
        Needs re-definition of the to_frontend method because the frontend receives full documents,
        whereas the LLM receives the chunks.
        So to_json() outputs the chunks (to the LLM)
        But the to_frontend() outputs the full documents and the chunk spans.
        """
        if self.metadata["chunked"] and not self.async_init_completed:
            raise Exception(
                "ERROR: Documents are chunked and DocumentRetrieval not initialized. "
                "Need to run .async_init(client_manager) to retrieve the full documents attached to the chunks. "
                "Otherwise set metadata['chunked'] to False"
            )

        payload = {
            "type": self.payload_type,
            "objects": (
                self.full_documents_to_json(mapping=True)
                if self.metadata["chunked"] and self.async_init_completed
                else self.to_json(mapping=True)
            ),
            "metadata": self.metadata,
        }

        if "code" in self.metadata:
            payload["code"] = self.metadata["code"]

        return {
            "type": self.frontend_type,
            "user_id": user_id,
            "conversation_id": conversation_id,
            "query_id": query_id,
            "id": self.frontend_type[:3] + "-" + str(uuid.uuid4()),
            "payload": payload,
        }


class Aggregation(Retrieval):
    def __init__(
        self,
        objects: list[dict],
        metadata: dict = {},
        name: str | None = None,
        **kwargs,
    ) -> None:
        if name is None and "collection_name" in metadata:
            name = metadata["collection_name"]
        elif name is None:
            name = "default"

        Retrieval.__init__(
            self,
            objects=objects,
            payload_type="aggregation",
            metadata=metadata,
            name=name,
            **kwargs,
        )

    def llm_parse(self) -> str:
        out = ""
        if "collection_name" in self.metadata:
            out += f"\nAggregated collection: '{self.metadata['collection_name']}'"

            count = {}
            if len(self.objects) > 0:
                for metric in self.objects[0]["collections"][0][
                    self.metadata["collection_name"]
                ]:
                    if metric != "ELYSIA_NUM_ITEMS":
                        num_values = len(
                            self.objects[0]["collections"][0][
                                self.metadata["collection_name"]
                            ][metric]["values"]
                        )
                        count[metric] = num_values

                for metric in count:
                    out += f", returned {count[metric]} values for {metric}."
                if any(count[metric] == 0 for metric in count):
                    out += f" For those metrics with 0 objects, judge the aggregation code that was created, and evaluate whether it was appropriate for the collection/metric, the user prompt, and the data available."
                    out += f" If it seemed innappropriate, you can choose to try again if you think it can still be completed (or there is more to do)."
            else:
                out += f" but no objects were returned."
                out += f" Since it had no objects, judge the aggregation code that was created, and evaluate whether it was appropriate for the collection, the user prompt, and the data available."
                out += f" If it seemed innappropriate, you can choose to try again if you think it can still be completed (or there is more to do)."

            if "groupby_name" in self.metadata:
                out += f"\nGrouped by property '{self.metadata['groupby_name']}'"
            if "metrics" in self.metadata:
                out += f"\nReturned metrics for property(-ies) '{self.metadata['metrics']}'"
            if "impossible" in self.metadata:
                if "collection_name" in self.metadata:
                    out += f"\nThis attempt at aggregating the collection: {self.metadata['collection_name']} was deemed impossible."
                out += f"The user prompt was: '{self.metadata['impossible']}', which the aggregation model deemed impossible to complete."
                if "impossible_reasoning" in self.metadata:
                    out += f"\nReasoning for impossibility: {self.metadata['impossible_reasoning']}"
            if "aggregation_output" in self.metadata:
                out += f"\nThe aggregation query used was:\n{self.metadata['aggregation_output']}"
        return out
