import json
import os
import unittest

from langchain_core.messages import AIMessage, ToolMessage

os.environ.setdefault("OPENAI_API_KEY", "test-key")

from gtm_agent.gtm_agent import VerifiedActionMiddleware


class VerifiedActionMiddlewareTests(unittest.TestCase):
    def setUp(self):
        self.middleware = VerifiedActionMiddleware()

    def run_guard(self, candidate, *tool_messages):
        result = self.middleware.after_model(
            {"messages": [*tool_messages, AIMessage(content=candidate)]},
            None,
        )
        return result["messages"][0].content if result else candidate

    def test_email_claim_without_send_tool_is_replaced(self):
        response = self.run_guard("The email was successfully sent to the prospect.")

        self.assertEqual(
            response,
            "The email was not sent because no successful send confirmation was produced.",
        )

    def test_empty_score_result_is_not_verified(self):
        score_result = ToolMessage(
            name="score_prospect",
            tool_call_id="score-1",
            content=json.dumps({"score": None, "max_score": 100}),
        )

        response = self.run_guard("The fit score is 90/100.", score_result)

        self.assertEqual(response, "No verified fit score was produced.")

    def test_failed_score_result_is_not_verified(self):
        score_result = ToolMessage(
            name="score_prospect",
            tool_call_id="score-2",
            content=json.dumps({"score": None, "error": "Cannot score without a valid offering."}),
        )

        response = self.run_guard("The fit score is 85/100.", score_result)

        self.assertEqual(response, "No verified fit score was produced.")

    def test_verified_send_passes_through_unchanged(self):
        send_result = ToolMessage(
            name="send_prospect_email",
            tool_call_id="send-1",
            content=json.dumps({
                "status": "sent",
                "message_id": "msg-123",
                "body": "Hello from the team.",
            }),
        )
        candidate = "The email was sent successfully. Body: Hello from the team."

        self.assertEqual(self.run_guard(candidate, send_result), candidate)


if __name__ == "__main__":
    unittest.main()
