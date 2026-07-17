from minirag.schemas import RouteResult
from minirag.routing import QueryRouter


def test_query_router_routes_comparison_question():
    router = QueryRouter()

    result = router.route_question("Compare Shakespeare and John Shakespeare.")

    assert result.route == "comparison"
    assert result.risk_level == "medium"


def test_query_router_routes_technical_safety_question():
    router = QueryRouter()

    result = router.route_question("Can I connect neutral and phase together?")

    assert result.route == "technical_safety"
    assert result.risk_level == "high"
