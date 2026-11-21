"""Free text -> query plan (aspects, lexicon tags, feelings, exclusions) for music.

Small embedding models are unreliable on affect words, so common mood / activity
words map to candidate tags. Tags missing from the current vocabulary are ignored
at query time, so the same lexicon works with or without Last.fm data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

VIBES: dict[str, list[str]] = {
    "chill|chilled|chillout|laid back|laid-back|laidback|easy listening|easygoing":
        ["chill", "mellow", "downtempo", "chillwave", "lounge", "trip hop", "lo-fi", "relaxing"],
    "relax|relaxing|relaxed|calm|peaceful|soothing|tranquil|serene|unwind":
        ["relaxing", "calm", "peaceful", "soothing", "ambient", "mellow", "chill", "new age"],
    "happy|cheerful|joyful|feel good|feel-good|sunny|bright|positive":
        ["happy", "feel good", "uplifting", "cheerful", "fun", "upbeat", "sunshine pop"],
    "uplifting|hopeful|inspiring|inspirational|motivating|motivational|empowering":
        ["uplifting", "hopeful", "inspiring", "anthemic", "euphoric", "feel good"],
    "upbeat|energetic|energy|hype|hyped|pumped|lively|high energy":
        ["energetic", "upbeat", "fast", "dance", "party", "powerful"],
    "sad|cry|crying|tearjerker|heartbreak|heartbroken|breakup|break-up":
        ["sad", "heartbreak", "melancholic", "emotional", "depressing", "bittersweet"],
    "melancholic|melancholy|wistful|bittersweet|blue|rainy|rain|gloomy":
        ["melancholic", "bittersweet", "sad", "wistful", "rainy day", "atmospheric"],
    "nostalgic|nostalgia|throwback|retro|old school|oldies":
        ["nostalgic", "nostalgia", "oldies", "retro", "classic rock", "80s", "70s"],
    "dreamy|ethereal|floaty|hazy|dreamlike":
        ["dreamy", "dream pop", "ethereal", "shoegaze", "ambient", "atmospheric", "chillwave"],
    "atmospheric|cinematic|epic|grand|soundtrack-like|majestic":
        ["atmospheric", "cinematic", "epic", "post-rock", "orchestral", "film score"],
    "dark|brooding|moody|ominous|eerie|haunting|creepy|spooky":
        ["dark", "brooding", "moody", "haunting", "eerie", "darkwave", "gothic", "dark ambient"],
    "angry|aggressive|rage|furious|heavy|intense|brutal":
        ["aggressive", "angry", "heavy", "intense", "metal", "hardcore", "industrial"],
    "romantic|love|lovey|date night|sensual|sexy|intimate":
        ["romantic", "love", "sensual", "sexy", "soul", "r&b", "slow jams", "love songs"],
    "groovy|funky|funk|groove|smooth":
        ["groovy", "funky", "funk", "smooth", "soul", "disco", "neo-soul"],
    "dance|dancing|party|club|clubbing|banger|bangers":
        ["party", "dance", "dance-pop", "house", "club", "edm", "electro house", "disco"],
    "trippy|psychedelic|hypnotic|weird|experimental|strange":
        ["psychedelic", "trippy", "hypnotic", "experimental", "weird", "krautrock", "neo-psychedelia"],
    "acoustic|unplugged|stripped|campfire|folky":
        ["acoustic", "folk", "indie folk", "singer-songwriter", "unplugged"],
    "quiet|soft|gentle|delicate|tender|lullaby":
        ["soft", "gentle", "quiet", "piano", "acoustic", "lullaby", "ambient"],
    "cozy|cosy|warm|comforting|comfort":
        ["warm", "cozy", "mellow", "acoustic", "indie folk", "soft", "jazz"],
    "catchy|earworm|singalong|sing-along|anthem|anthems":
        ["catchy", "anthemic", "pop", "singalong", "power pop"],
    "lonely|alone|isolated|solitude":
        ["lonely", "melancholic", "sad", "atmospheric", "slowcore"],
    # activities / settings
    "study|studying|focus|concentrate|concentration|work|working|coding|homework|reading":
        ["study", "focus", "lo-fi", "ambient", "instrumental hip hop", "post-rock", "minimal", "modern classical"],
    "sleep|sleeping|bedtime|fall asleep|drift off":
        ["sleep", "ambient", "calm", "drone", "soft", "piano", "new age"],
    "workout|gym|running|run|lifting|training|cardio":
        ["workout", "energetic", "running", "hip-hop", "edm", "hard rock", "pump up"],
    "driving|drive|road trip|roadtrip|highway|cruising":
        ["driving", "road trip", "synthwave", "classic rock", "heartland rock", "upbeat"],
    "night|late night|midnight|after hours|3am|nocturnal":
        ["night", "late night", "nocturnal", "synthwave", "trip hop", "downtempo", "dark"],
    "summer|beach|sunshine|tropical|vacation|poolside":
        ["summer", "beach", "tropical", "sunshine pop", "reggae", "surf rock", "feel good"],
    "winter|snow|cold|christmas|holiday|festive":
        ["winter", "christmas", "cold", "atmospheric", "folk"],
    "autumn|fall|rainy day|sunday|sunday morning|lazy":
        ["autumn", "rainy day", "sunday morning", "mellow", "indie folk", "jazz", "acoustic"],
    "morning|wake up|coffee|breakfast":
        ["morning", "coffee", "acoustic", "indie pop", "jazz", "feel good"],
    "dinner|cooking|background|cafe|coffee shop|lounge":
        ["lounge", "jazz", "bossa nova", "background", "smooth jazz", "downtempo"],
    "gaming|video game|8-bit|chiptune":
        ["video game music", "chiptune", "electronic", "synthwave"],
}

FEELINGS: dict[str, tuple[list[str], list[str], list[str]]] = {
    "sad|down|depressed|blue|low|miserable|heartbroken|gloomy": (
        ["uplifting", "happy", "feel good", "hopeful", "warm"],
        ["sad", "melancholic", "heartbreak", "bittersweet"],
        ["depressing", "sad", "dark"]),
    "stressed|anxious|overwhelmed|nervous|tense|burned out|burnt out": (
        ["calm", "relaxing", "ambient", "soothing", "mellow"],
        ["intense", "dark", "industrial", "anxious"],
        ["aggressive", "intense", "heavy", "noise"]),
    "tired|exhausted|sleepy|drained": (
        ["mellow", "chill", "soft", "warm"],
        ["mellow", "soft", "ambient", "sleep"], ["aggressive", "heavy"]),
    "lonely|alone|isolated": (
        ["warm", "uplifting", "feel good", "soul"],
        ["lonely", "melancholic", "sad", "slowcore"], ["depressing"]),
    "bored|restless": (
        ["energetic", "upbeat", "fun", "catchy", "dance"],
        ["energetic", "upbeat", "fun", "catchy"], []),
    "angry|mad|frustrated|pissed|furious": (
        ["aggressive", "punk", "metal", "cathartic", "intense"],
        ["aggressive", "angry", "metal", "hardcore"], []),
    "happy|great|excited|joyful|good|amazing|ecstatic": (
        ["happy", "upbeat", "feel good", "dance", "fun"],
        ["happy", "upbeat", "feel good", "dance", "fun"], []),
    "nostalgic|sentimental": (
        ["nostalgic", "nostalgia", "oldies", "retro"],
        ["nostalgic", "nostalgia", "oldies", "retro"], []),
    "in love|romantic|lovey": (
        ["romantic", "love", "love songs", "soul"],
        ["romantic", "love", "love songs", "soul"], []),
    "focused|productive|motivated": (
        ["focus", "study", "energetic", "uplifting"],
        ["focus", "study", "energetic", "uplifting"], []),
}

_FILLER = re.compile(
    r"\b(i\s+(?:want|need|would like)|i'?d like|i'?m\s+(?:in the mood|looking)\s+for|in the mood for|"
    r"looking for|want|wanna|need|give me|play me|play|put on|queue|recommend(?: me)?|suggest(?: me)?|"
    r"something|anything|some|songs?|tracks?|music|tunes|a playlist|playlist|stuff|vibes?|"
    r"that (?:feels?|is|sounds?|makes? me feel)|to listen to|for me|right now|tonight|please)\b", re.I)
_FEEL = re.compile(
    r"\b(?:i'?m|i am|im|feeling|i feel|feel)\s+(?:so\s+|really\s+|very\s+|kinda\s+|a bit\s+|pretty\s+|super\s+|quite\s+)?"
    r"([a-z]+(?:\s+(?:out|in love))?)", re.I)
_NEG = re.compile(
    r"\b(?:no|not|without|nothing|avoid|never|don'?t want(?: anything)?|but not)\s+"
    r"(?:too\s+|very\s+|so\s+|any\s+|overly\s+)?([a-z][a-z' &-]*?)(?=\s*(?:,|\.|;|!|\band\b|\bbut\b|\bor\b|$))",
    re.I)
_SPLIT = re.compile(r",|;|\.|\band\b|\bbut\b|\bwith\b|\bplus\b|&|/|\bfor\b", re.I)


def _compile(lex: dict) -> list:
    return [(re.compile(r"\b(?:" + "|".join(re.escape(a) for a in sorted(k.split("|"), key=len, reverse=True))
                        + r")\b", re.I), v) for k, v in lex.items()]


_VIBES = _compile(VIBES)
_FEELINGS = _compile(FEELINGS)


@dataclass
class QueryPlan:
    text: str
    aspects: list[tuple[str, list[str]]] = field(default_factory=list)
    feeling: str | None = None
    feeling_tags: list[str] = field(default_factory=list)
    excludes: list[str] = field(default_factory=list)
    exclude_tags: dict[str, float] = field(default_factory=dict)


def lexicon_tags(phrase: str) -> list[str]:
    out: list[str] = []
    for pat, tags in _VIBES:
        if pat.search(phrase):
            out += tags
    return out


def _add(d: dict[str, float], tags: list[str], w: float = 1.0) -> None:
    for t in tags:
        d[t] = d.get(t, 0.0) + w / len(tags)


def plan(text: str, mood_mode: str = "lift") -> QueryPlan:
    p = QueryPlan(text=text)
    rest = text
    p.excludes = [m.group(1).strip() for m in _NEG.finditer(rest)]
    rest = _NEG.sub(" , ", rest)
    p.excludes += [w[1:] for w in re.findall(r"(?<!\S)-[a-z][\w&-]*", rest, re.I)]
    rest = re.sub(r"(?<!\S)-[a-z][\w&-]*", " , ", rest, flags=re.I)
    for ex in p.excludes:
        _add(p.exclude_tags, lexicon_tags(ex) or [ex.lower()])

    for m in _FEEL.finditer(rest):
        word = m.group(1).lower()
        for pat, (lift, match, avoid) in _FEELINGS:
            if pat.fullmatch(word):
                p.feeling = word
                p.feeling_tags = lift if mood_mode == "lift" else match
                if mood_mode == "lift" and avoid:
                    _add(p.exclude_tags, avoid, 0.6)
                rest = rest.replace(m.group(0), " , ")
                break
        if p.feeling:
            break

    rest = _FILLER.sub(" ", rest)
    for chunk in _SPLIT.split(rest):
        chunk = re.sub(r"\s+", " ", chunk).strip(" ,.;!?-")
        if len(chunk) < 2 or chunk.lower() in {"a", "an", "the", "me", "really", "kind of", "like", "my"}:
            continue
        p.aspects.append((chunk, lexicon_tags(chunk)))
    return p
