from app.services.rag_engine import RAGEngineService

def test_faithfulness_score_calculation():
    """Tests overlap grounding math for faithfulness calculation."""
    answer = "The Segment Anything Model uses an image encoder and mask decoder."
    context = "SAM consists of a heavy image encoder and a promptable mask decoder."
    
    score = RAGEngineService.compute_faithfulness_score(answer, context)
    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0
    assert score >= 0.5