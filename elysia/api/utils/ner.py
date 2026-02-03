from multi_rake import Rake
import re
import asyncio


def _get_entities_with_spans(text: str):
    rake = Rake()
    keywords = rake.apply(text)

    results = []

    for keyword, score in keywords:
        pattern = re.escape(keyword)
        for match in re.finditer(pattern, text, re.IGNORECASE):
            results.append(
                {
                    "text": match.group(),
                    "start": match.start(),
                    "end": match.end(),
                    "score": score,
                }
            )

    results.sort(key=lambda x: x["start"])

    return results


async def named_entity_recognition(text: str):
    """
    Performs Named Entity Recognition using multi_rake.
    Returns a list of entities with their labels, start and end positions.
    """
    try:
        entities = await asyncio.to_thread(_get_entities_with_spans, text)
        out = {"text": text, "entity_spans": [], "noun_spans": [], "error": ""}

        for ent in entities:
            out["entity_spans"].append((ent["start"], ent["end"]))

        return out

    except Exception as e:
        return {
            "text": text,
            "entity_spans": [],
            "error": str(e),
        }
