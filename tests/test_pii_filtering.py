import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


SENSITIVE_FIELDS = ("tax_id", "date_of_birth", "card_on_file", "credit_check_ref")


class PiiFilteringTests(unittest.TestCase):
    def setUp(self):
        data_service._PROFILES.clear()

    def test_prospect_tools_and_scoring_prompt_exclude_billing_qualification(self):
        prospect_id = "LEAD-39002"
        record = data_service.get_prospect_record(prospect_id)

        contact = gtm_agent.get_prospect.invoke({"prospect_id": prospect_id})
        self.assertNotIn("billing_qualification", json.dumps(contact))
        for field in SENSITIVE_FIELDS:
            self.assertNotIn(field, json.dumps(contact))

        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": prospect_id})
        self.assertNotIn("billing_qualification", json.dumps(profile))
        for field in SENSITIVE_FIELDS:
            self.assertNotIn(field, json.dumps(profile))
        self.assertNotIn("billing_qualification", json.dumps(data_service._PROFILES[prospect_id]))

        result = Mock()
        result.model_dump.return_value = {"score": 80}
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.return_value = result
            gtm_agent.score_prospect.invoke({
                "prospect_profile": {
                    **record,
                    "prospect_id": prospect_id,
                    "billing_qualification": record["billing_qualification"],
                },
                "offering": data_service.get_offering("OFFER-10006"),
            })

        prompt = scoring_llm.invoke.call_args.args[0][1]["content"]
        self.assertNotIn("billing_qualification", prompt)
        for field in SENSITIVE_FIELDS:
            self.assertNotIn(field, prompt)


if __name__ == "__main__":
    unittest.main()
