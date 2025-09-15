from elysia.config import nlp


def named_entity_recognition(text: str):
    """
    Performs Named Entity Recognition using spaCy.
    Returns a list of entities with their labels, start and end positions.
    """
    try:
        doc = nlp(text)
        out = {"text": text, "entity_spans": [], "noun_spans": [], "error": ""}

        for ent in doc.ents:
            out["entity_spans"].append((ent.start_char, ent.end_char))

        # Get noun spans
        for token in doc:
            if token.pos_ == "NOUN":
                span = doc[token.i : token.i + 1]
                out["noun_spans"].append((span.start_char, span.end_char))

        return out

    except Exception as e:
        return {
            "text": text,
            "entity_spans": [],
            "noun_spans": [],
            "error": str(e),
        }
