from naijalingo_asr import transcribe

def test_import_only():
    assert callable(transcribe)
