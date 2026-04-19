import hashlib
from datetime import datetime
from sqlalchemy import select
from core.database import AsyncSessionLocal
from core.models import ConceptMastery

_DOMAIN_SIGNATURES = {
    "math": {"wave": "sine", "base_freq": 440.0, "interval_ratio": 1.5, "rhythm": "precise"},
    "physics": {"wave": "triangle", "base_freq": 396.0, "interval_ratio": 1.414, "rhythm": "wave"},
    "biology": {"wave": "sine", "base_freq": 528.0, "interval_ratio": 1.618, "rhythm": "organic"},
    "history": {"wave": "sawtooth", "base_freq": 333.0, "interval_ratio": 1.333, "rhythm": "march"},
    "philosophy": {"wave": "square", "base_freq": 417.0, "interval_ratio": 1.732, "rhythm": "contemplative"},
    "computer": {"wave": "square", "base_freq": 480.0, "interval_ratio": 2.0, "rhythm": "digital"},
    "chemistry": {"wave": "triangle", "base_freq": 432.0, "interval_ratio": 1.25, "rhythm": "molecular"},
    "economics": {"wave": "sawtooth", "base_freq": 360.0, "interval_ratio": 1.2, "rhythm": "cyclical"},
    "default": {"wave": "sine", "base_freq": 440.0, "interval_ratio": 1.5, "rhythm": "neutral"},
}


def _detect_domain(concept: str) -> str:
    concept_lower = concept.lower()
    for domain in _DOMAIN_SIGNATURES:
        if domain in concept_lower:
            return domain
    keywords = {
        "math": ["equation", "algebra", "calculus", "geometry", "proof", "theorem", "number"],
        "physics": ["force", "energy", "quantum", "relativity", "wave", "particle", "motion"],
        "biology": ["cell", "dna", "evolution", "gene", "protein", "organism", "species"],
        "history": ["war", "empire", "revolution", "civilization", "century", "dynasty"],
        "philosophy": ["ethics", "logic", "epistemology", "ontology", "consciousness", "truth"],
        "computer": ["algorithm", "data", "function", "code", "neural", "machine", "software"],
        "chemistry": ["atom", "molecule", "bond", "reaction", "element", "compound"],
        "economics": ["market", "supply", "demand", "trade", "price", "inflation", "capital"],
    }
    for domain, kws in keywords.items():
        if any(kw in concept_lower for kw in kws):
            return domain
    return "default"


def _concept_hash_to_params(concept: str) -> dict:
    h = int(hashlib.md5(concept.encode()).hexdigest(), 16)
    return {
        "detune": (h % 100) - 50,
        "attack": 0.1 + (h % 10) * 0.05,
        "release": 0.5 + (h % 20) * 0.1,
        "reverb": 0.2 + (h % 5) * 0.1,
    }


class SymphonyEngine:
    async def generate_motif(self, concept: str) -> dict:
        domain = _detect_domain(concept)
        sig = _DOMAIN_SIGNATURES[domain]
        params = _concept_hash_to_params(concept)

        base_freq = sig["base_freq"]
        ratio = sig["interval_ratio"]

        motif = {
            "concept": concept,
            "domain": domain,
            "tone_js": {
                "oscillator": {
                    "type": sig["wave"],
                    "frequency": base_freq,
                    "detune": params["detune"],
                },
                "envelope": {
                    "attack": params["attack"],
                    "decay": 0.3,
                    "sustain": 0.6,
                    "release": params["release"],
                },
                "reverb": {"wet": params["reverb"]},
                "notes": [
                    {"freq": round(base_freq, 2), "duration": "4n", "time": 0},
                    {"freq": round(base_freq * ratio, 2), "duration": "4n", "time": 0.5},
                    {"freq": round(base_freq * ratio * ratio, 2), "duration": "2n", "time": 1.0},
                ],
                "rhythm": sig["rhythm"],
                "bpm": 80 + (int(hashlib.md5(concept.encode()).hexdigest(), 16) % 40),
            },
            "generated_at": datetime.utcnow().isoformat(),
        }

        return motif

    async def store_motif(self, session_id: str, concept: str, motif: dict) -> None:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery).where(
                    ConceptMastery.session_id == session_id,
                    ConceptMastery.concept == concept,
                )
            )
            cm = result.scalar_one_or_none()
            if cm is None:
                cm = ConceptMastery(
                    session_id=session_id,
                    concept=concept,
                    tide_data={"motif": motif},
                )
                db.add(cm)
            else:
                existing_tide = dict(cm.tide_data or {})
                existing_tide["motif"] = motif
                cm.tide_data = existing_tide
            await db.commit()

    async def harmonize(self, motif_a: dict, motif_b: dict) -> dict:
        freq_a = motif_a.get("tone_js", {}).get("oscillator", {}).get("frequency", 440.0)
        freq_b = motif_b.get("tone_js", {}).get("oscillator", {}).get("frequency", 440.0)

        ratio = freq_b / freq_a if freq_a > 0 else 1.0
        harmony_freq = (freq_a + freq_b) / 2

        HARMONY_TYPES = {
            range(0, 10): "unison",
            range(10, 20): "minor_second",
            range(20, 35): "major_third",
            range(35, 50): "perfect_fifth",
            range(50, 70): "major_seventh",
            range(70, 100): "octave",
        }

        ratio_pct = int(abs(ratio - 1.0) * 100)
        harmony_type = "consonant"
        for r, name in HARMONY_TYPES.items():
            if ratio_pct in r:
                harmony_type = name
                break

        return {
            "concept_a": motif_a.get("concept"),
            "concept_b": motif_b.get("concept"),
            "harmony_type": harmony_type,
            "harmony_frequency": round(harmony_freq, 2),
            "cognitive_link": f"Understanding {motif_a.get('concept')} resonates with {motif_b.get('concept')} through {harmony_type}",
            "tone_js": {
                "oscillator": {"type": "sine", "frequency": round(harmony_freq, 2)},
                "envelope": {"attack": 0.3, "decay": 0.5, "sustain": 0.7, "release": 1.0},
                "notes": [
                    {"freq": round(freq_a, 2), "duration": "2n", "time": 0},
                    {"freq": round(freq_b, 2), "duration": "2n", "time": 0.25},
                    {"freq": round(harmony_freq, 2), "duration": "1n", "time": 0.5},
                ],
            },
        }

    async def get_symphony(self, session_id: str) -> list:
        async with AsyncSessionLocal() as db:
            result = await db.execute(
                select(ConceptMastery)
                .where(ConceptMastery.session_id == session_id)
                .order_by(ConceptMastery.mastery_score.desc())
            )
            masteries = result.scalars().all()

        symphony = []
        for cm in masteries:
            tide = cm.tide_data or {}
            if "motif" in tide:
                symphony.append(tide["motif"])
            else:
                motif = await self.generate_motif(cm.concept)
                symphony.append(motif)

        return symphony


symphony_engine = SymphonyEngine()
