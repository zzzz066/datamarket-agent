from __future__ import annotations

import unittest

from agents.base import (
    ActionParser,
    AgentActionParseError,
    OverbidBuyerAgent,
    RuleBasedPlatformAgent,
    ShadeBuyerAgent,
    TruthfulBuyerAgent,
    LLMAgent,
    fallback_action,
)


class FakeClient:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls = []

    def complete(self, messages, *, model, temperature=0.2, response_format=None):
        self.calls.append({
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "response_format": response_format,
        })
        return self.content


class ActionParserTest(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ActionParser()
        self.platform_obs = {"候选价格": [0.1, 0.5, 1.0], "MWU推荐价格": 0.5}
        self.buyer_obs = {"我的估值_mu": 0.8, "当前平台价格": 0.6}

    def test_parse_plain_platform_json(self) -> None:
        self.assertEqual(
            self.parser.parse('{"reasoning":"稳健", "价格": 0.7}', obs=self.platform_obs),
            {"价格": 0.7},
        )

    def test_parse_fenced_buyer_json(self) -> None:
        raw = '''```json\n{"reasoning":"诚实报价", "报价": "0.8"}\n```'''
        self.assertEqual(self.parser.parse(raw, obs=self.buyer_obs), {"报价": 0.8})

    def test_parse_text_with_nested_action_and_alias(self) -> None:
        raw = '我会低报一点。 {"action": {"bid": 0.52}, "note": "shade"}'
        self.assertEqual(self.parser.parse(raw, obs=self.buyer_obs), {"报价": 0.52})

    def test_platform_price_clamped_to_candidate_bounds(self) -> None:
        self.assertEqual(
            self.parser.parse('{"price": 9.0}', obs=self.platform_obs),
            {"价格": 1.0},
        )

    def test_invalid_payload_raises(self) -> None:
        with self.assertRaises(AgentActionParseError):
            self.parser.parse('no json here', obs=self.buyer_obs)


class RuleAgentTest(unittest.TestCase):
    def test_rule_agents(self) -> None:
        platform_obs = {"MWU推荐价格": 0.45}
        buyer_obs = {"我的估值_mu": 1.0}
        self.assertEqual(RuleBasedPlatformAgent().act(platform_obs), {"价格": 0.45})
        self.assertEqual(TruthfulBuyerAgent().act(buyer_obs), {"报价": 1.0})
        self.assertEqual(ShadeBuyerAgent(0.6).act(buyer_obs), {"报价": 0.6})
        self.assertEqual(OverbidBuyerAgent(1.2).act(buyer_obs), {"报价": 1.2})
        self.assertEqual(fallback_action(buyer_obs), {"报价": 1.0})


class LLMAgentTest(unittest.TestCase):
    def test_llm_agent_uses_client_and_parser(self) -> None:
        client = FakeClient('{"reasoning":"沿用基线", "报价": 0.75}')
        agent = LLMAgent(client=client, model="fake-model", role="买家", retry_sleep=0)
        action = agent.act({"当前轮次": 1, "我的估值_mu": 0.8, "当前平台价格": 0.6})
        self.assertEqual(action, {"报价": 0.75})
        self.assertEqual(client.calls[0]["model"], "fake-model")
        self.assertEqual(client.calls[0]["response_format"], {"type": "json_object"})


if __name__ == "__main__":
    unittest.main()
