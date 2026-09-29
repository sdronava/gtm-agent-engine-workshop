import os

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent import data_service
from gtm_agent.gtm_agent import build_prospect_profile


def test_update_prospect_info_persists_and_invalidates_profile_cache():
    prospect_id = "LEAD-71001"
    data_service._PROFILES.pop(prospect_id, None)

    build_prospect_profile.invoke({"prospect_id": prospect_id})
    data_service.update_prospect_info(prospect_id, "Kafka")

    assert "Kafka" in data_service.fetch_tech_stack(prospect_id)
    profile = build_prospect_profile.invoke({"prospect_id": prospect_id})
    assert "Kafka" in profile["prospect_profile"]["tech_stack"]
