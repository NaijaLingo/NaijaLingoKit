MODEL_REPO_BY_LANG = {
    "yo": "chukypedro/ct2_whisper_yo",
    "ig": "chukypedro/ct2_whisper_ig",
    "ha": "chukypedro/ct2_whisper_ha",
    "en": "chukypedro/ct2_whisper_ng-en",
}

SUPPORTED_LANGUAGES = set(MODEL_REPO_BY_LANG.keys())


def get_repo_for_language(language_code: str) -> str:
    code = (language_code or "").strip().lower()
    # normalize common names/aliases
    aliases = {
        "yoruba": "yo",
        "igbo": "ig",
        "hausa": "ha",
        "english": "en",
        "nigerian-english": "en",
        "nigerian english": "en",
        "ng-en": "en",
        "en-ng": "en",
    }
    code = aliases.get(code, code)
    if code not in MODEL_REPO_BY_LANG:
        supported = ", ".join(sorted(MODEL_REPO_BY_LANG))
        raise ValueError(f"Unsupported language {language_code}. Supported: {supported}")
    return MODEL_REPO_BY_LANG[code]
