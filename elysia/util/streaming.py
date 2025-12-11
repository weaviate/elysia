from enum import Enum
from typing import Literal
from dataclasses import dataclass
import uuid


class TextParserState(Enum):
    IDLE = "idle"
    LOOKING_FOR_ARRAY = "looking_for_array"
    IN_ARRAY = "in_array"
    IN_TEXT = "in_text"
    IN_REF_IDS = "in_ref_ids"
    IN_REF_ID_VALUE = "in_ref_id_value"


@dataclass
class StreamedPayload:
    type: Literal["text", "citation", "metadata", "end"]
    chunk: str | dict | None
    index: int | None


@dataclass
class TextChunkPayload(StreamedPayload):
    type: Literal["text"] = "text"
    chunk: str = ""
    index: int = 0


@dataclass
class TextCitationPayload(StreamedPayload):
    type: Literal["citation"] = "citation"
    chunk: str = ""  # a full ref_id
    index: int = 0


class StreamedParser:
    def __init__(self):
        pass

    def feed(self, chunk: str) -> list[StreamedPayload]:
        raise NotImplementedError("Subclasses must implement this method")

    def reset(self):
        pass


class StreamedTextWithCitations(StreamedParser):

    def __init__(self):
        StreamedParser.__init__(self)
        self.buffer = ""
        self.state = TextParserState.IDLE
        self.current_index = -1  # Index in the list of ListTextWithCitation objects
        self.current_ref_id = ""  # Buffer for building ref_id strings
        self.debug_full_string = ""

    def feed(self, chunk: str) -> list[TextChunkPayload | TextCitationPayload]:
        """
        Feed a raw JSON chunk and yield structured payloads.
        """
        streamed_payloads = []
        self.buffer += chunk
        self.debug_full_string += chunk

        while self.buffer:
            if self.state == TextParserState.IDLE:
                # Look for "cited_text" field to enter the array
                cited_text_field = self.buffer.find('"cited_text"')
                if cited_text_field != -1:
                    # Find the colon and opening bracket
                    colon_pos = self.buffer.find(":", cited_text_field)
                    if colon_pos != -1:
                        self.state = TextParserState.LOOKING_FOR_ARRAY
                        self.buffer = self.buffer[colon_pos + 1 :]
                        continue
                    break

                # No recognizable pattern, clear buffer keeping last chars
                if len(self.buffer) > 20:
                    self.buffer = self.buffer[-20:]
                break

            elif self.state == TextParserState.LOOKING_FOR_ARRAY:
                # Look for the opening bracket of the array
                bracket_pos = self.buffer.find("[")
                if bracket_pos != -1:
                    self.state = TextParserState.IN_ARRAY
                    self.buffer = self.buffer[bracket_pos + 1 :]
                    continue
                break

            elif self.state == TextParserState.IN_ARRAY:
                # Find positions of all relevant markers
                obj_start = self.buffer.find("{")
                array_end = self.buffer.find("]")
                text_field = self.buffer.find('"text"')
                ref_field = self.buffer.find('"ref_ids"')
                obj_end = self.buffer.find("}")

                # Determine the earliest relevant position
                # (we need to process things in order of appearance)
                positions = []
                if obj_start != -1:
                    positions.append(("obj_start", obj_start))
                if array_end != -1:
                    positions.append(("array_end", array_end))
                if text_field != -1:
                    positions.append(("text_field", text_field))
                if ref_field != -1:
                    positions.append(("ref_field", ref_field))
                if obj_end != -1:
                    positions.append(("obj_end", obj_end))

                if not positions:
                    # Nothing found - need more data
                    break

                # Sort by position and handle the earliest one
                positions.sort(key=lambda x: x[1])
                earliest, earliest_pos = positions[0]

                if earliest == "array_end":
                    # Array ended
                    self.state = TextParserState.IDLE
                    self.buffer = self.buffer[array_end + 1 :]
                    continue

                if earliest == "obj_start":
                    # New object - increment index
                    self.current_index += 1
                    self.buffer = self.buffer[obj_start + 1 :]
                    continue

                if earliest == "text_field":
                    # Found "text" field
                    colon_pos = self.buffer.find(":", text_field)
                    if colon_pos != -1:
                        quote_pos = self.buffer.find('"', colon_pos + 1)
                        if quote_pos != -1:
                            self.state = TextParserState.IN_TEXT
                            self.buffer = self.buffer[quote_pos + 1 :]
                            continue
                    break

                if earliest == "ref_field":
                    # Found "ref_ids" field
                    bracket_pos = self.buffer.find("[", ref_field)
                    if bracket_pos != -1:
                        self.state = TextParserState.IN_REF_IDS
                        self.buffer = self.buffer[bracket_pos + 1 :]
                        continue
                    break

                if earliest == "obj_end":
                    # End of current object - skip past it
                    self.buffer = self.buffer[obj_end + 1 :]
                    continue

                # Shouldn't reach here, but break just in case
                break

            elif self.state == TextParserState.IN_TEXT:
                # Stream text content until we hit an unescaped closing quote
                i = 0
                text_chunk = ""
                while i < len(self.buffer):
                    char = self.buffer[i]
                    if char == "\\":
                        if i + 1 < len(self.buffer):
                            escaped_char = self.buffer[i + 1]
                            if escaped_char == "n":
                                text_chunk += "\n"
                            elif escaped_char == "t":
                                text_chunk += "\t"
                            elif escaped_char == '"':
                                text_chunk += '"'
                            elif escaped_char == "\\":
                                text_chunk += "\\"
                            else:
                                text_chunk += escaped_char
                            i += 2
                            continue
                        else:
                            # Incomplete escape - keep the backslash in buffer for next chunk
                            if text_chunk:
                                streamed_payloads.append(
                                    TextChunkPayload(
                                        chunk=text_chunk, index=self.current_index
                                    )
                                )
                            self.buffer = self.buffer[i:]  # Keep from backslash onwards
                            break
                    elif char == '"':
                        if text_chunk:
                            streamed_payloads.append(
                                TextChunkPayload(
                                    chunk=text_chunk, index=self.current_index
                                )
                            )
                        self.buffer = self.buffer[i + 1 :]
                        self.state = TextParserState.IN_ARRAY
                        break
                    else:
                        text_chunk += char
                        i += 1
                else:
                    # Consumed entire buffer without finding end quote
                    if text_chunk:
                        streamed_payloads.append(
                            TextChunkPayload(chunk=text_chunk, index=self.current_index)
                        )
                    self.buffer = ""

                # Always break outer loop after processing IN_TEXT
                # (either we found closing quote and changed state, or we need more data)
                if self.state == TextParserState.IN_TEXT:
                    break

            elif self.state == TextParserState.IN_REF_IDS:
                quote_pos = self.buffer.find('"')
                bracket_end = self.buffer.find("]")

                if bracket_end != -1 and (quote_pos == -1 or bracket_end < quote_pos):
                    self.state = TextParserState.IN_ARRAY  # Back to IN_ARRAY
                    self.buffer = self.buffer[bracket_end + 1 :]
                    continue

                if quote_pos != -1:
                    self.state = TextParserState.IN_REF_ID_VALUE
                    self.current_ref_id = ""
                    self.buffer = self.buffer[quote_pos + 1 :]
                    continue

                break

            elif self.state == TextParserState.IN_REF_ID_VALUE:
                i = 0
                while i < len(self.buffer):
                    char = self.buffer[i]
                    if char == "\\":
                        if i + 1 < len(self.buffer):
                            self.current_ref_id += self.buffer[i + 1]
                            i += 2
                            continue
                        # Incomplete escape - keep buffer from here
                        self.buffer = self.buffer[i:]
                        break
                    elif char == '"':
                        streamed_payloads.append(
                            TextCitationPayload(
                                chunk=self.current_ref_id, index=self.current_index
                            )
                        )
                        self.current_ref_id = ""
                        self.buffer = self.buffer[i + 1 :]
                        self.state = TextParserState.IN_REF_IDS
                        break
                    else:
                        self.current_ref_id += char
                        i += 1
                else:
                    # Consumed entire buffer without finding end quote
                    self.buffer = ""

                # Only break outer loop if we're still in IN_REF_ID_VALUE (need more data)
                if self.state == TextParserState.IN_REF_ID_VALUE:
                    break

        return streamed_payloads

    def reset(self):
        self.buffer = ""
        self.state = TextParserState.IDLE
        self.current_index = -1
        self.current_ref_id = ""


class StreamedString(StreamedParser):
    def __init__(self):
        StreamedParser.__init__(self)

    def feed(self, chunk: str) -> list[StreamedPayload]:
        return [StreamedPayload(type="text", chunk=chunk, index=0)]


class StreamedDict(StreamedParser):
    def __init__(self):
        StreamedParser.__init__(self)

    def feed(self, chunk: dict) -> list[StreamedPayload]:
        return [StreamedPayload(type="metadata", chunk=chunk, index=0)]


class StreamedEnd(StreamedParser):
    def __init__(self):
        StreamedParser.__init__(self)

    def feed(self, chunk: None) -> list[StreamedPayload]:
        return [StreamedPayload(type="end", chunk=None, index=None)]


class StreamEndMarker:
    pass


class StreamedParserFactory:
    def __init__(self):
        from elysia.tools.text.objects import ListTextWithCitation

        self.factory = {
            ListTextWithCitation: StreamedTextWithCitations,
            str: StreamedString,
            dict: StreamedDict,
            StreamEndMarker: StreamedEnd,
        }
        self.parsers = {}
        self.parser_ids = {}

    def get_parser(
        self, field_type: type, field_name: str
    ) -> tuple[StreamedParser, str]:
        if field_name not in self.parsers:
            self.parsers[field_name] = {}
            self.parser_ids[field_name] = str(uuid.uuid4())

        if field_type not in self.parsers[field_name]:
            self.parsers[field_name][field_type] = self.factory[field_type]()

        return self.parsers[field_name][field_type], self.parser_ids[field_name]

    def reset(self):
        self.parsers = {}
        self.parser_ids = {}
