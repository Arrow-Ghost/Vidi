"""
Maps the words a user actually says onto the class names the detector emits.

A user says "my phone", "the table", "a bag" — COCO says "cell phone",
"dining table", "handbag". Without this layer, find_object("phone") returns
"not found" while a phone sits in plain view, which reads to the user as a
broken system rather than a vocabulary mismatch.

Anything NOT resolvable here is not silently dropped: it is reported as
out-of-vocabulary so the agent can say it cannot look for that specific
thing, rather than claiming the object is absent.
"""

from __future__ import annotations

from typing import List, Optional

from app.perception.object_detector import RELEVANT_CLASSES

# One spoken word may legitimately map to several detector classes: "bag"
# really could be any of three COCO classes, and the honest response is to
# search all of them and report what was found.
SYNONYMS: dict[str, List[str]] = {
    "phone": ["cell phone"], "mobile": ["cell phone"], "cellphone": ["cell phone"],
    "smartphone": ["cell phone"], "telephone": ["cell phone"],
    "table": ["dining table"], "desk": ["dining table"], "counter": ["dining table"],
    "bag": ["backpack", "handbag", "suitcase"], "purse": ["handbag"],
    "rucksack": ["backpack"], "knapsack": ["backpack"], "luggage": ["suitcase"],
    "bagpack": ["backpack"],  # common misspelling/mishearing
    "computer": ["laptop"], "notebook": ["laptop"], "pc": ["laptop"],
    "water bottle": ["bottle"], "waterbottle": ["bottle"], "flask": ["bottle"],
    "glass": ["cup", "wine glass"], "mug": ["cup"], "tumbler": ["cup"],
    "seat": ["chair"], "stool": ["chair"], "sofa": ["couch"], "settee": ["couch"],
    "people": ["person"], "someone": ["person"], "man": ["person"],
    "woman": ["person"], "human": ["person"], "guy": ["person"],
    "anybody": ["person"], "anyone": ["person"],
    "tv": ["tv"], "television": ["tv"], "monitor": ["tv"], "screen": ["tv"],
    "plant": ["potted plant"], "bin": ["potted plant"],
    "food": ["banana", "apple", "sandwich", "bowl"],
    "fruit": ["banana", "apple"], "plate": ["bowl"], "dish": ["bowl"],
    "cycle": ["bicycle"], "bike": ["bicycle", "motorcycle"],
    "vehicle": ["car", "bus", "truck", "motorcycle"],
    "automobile": ["car"], "van": ["truck"], "lorry": ["truck"],
    "fridge": ["refrigerator"], "clock": ["clock"], "watch": ["clock"],
}

# Things a user will plausibly ask for that this detector genuinely cannot
# find. Listing them explicitly lets the agent say "I can't reliably look for
# keys" instead of the much worse "I don't see your keys" — which the user
# would reasonably hear as "your keys aren't there".
KNOWN_UNSUPPORTED = {
    "keys", "key", "wallet", "door", "doors", "entrance", "exit", "stairs",
    "staircase", "steps", "sign", "signs", "id card", "id", "card", "curb",
    "kerb", "crosswalk", "pole", "wall", "corridor", "hallway", "railing",
}


def resolve(spoken: str) -> Optional[List[str]]:
    """
    Returns the detector classes to search for, or None if the phrase is not
    something this detector can find. Never guesses.
    """
    if not spoken:
        return None
    text = spoken.strip().lower()
    for prefix in ("my ", "the ", "a ", "an ", "some ", "that ", "this "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    text = text.rstrip("?.!,").strip()

    if text in RELEVANT_CLASSES:
        return [text]
    if text in SYNONYMS:
        return [c for c in SYNONYMS[text] if c in RELEVANT_CLASSES] or None
    # Singularize a trailing plural ("bottles" -> "bottle").
    if text.endswith("s"):
        singular = text[:-1]
        if singular in RELEVANT_CLASSES:
            return [singular]
        if singular in SYNONYMS:
            return [c for c in SYNONYMS[singular] if c in RELEVANT_CLASSES] or None
    return None


def is_known_unsupported(spoken: str) -> bool:
    text = (spoken or "").strip().lower().rstrip("?.!,")
    for prefix in ("my ", "the ", "a ", "an ", "some "):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text in KNOWN_UNSUPPORTED
