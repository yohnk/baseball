import asyncio
import json
import unittest
from typing import Awaitable

from playground.api.API import API


class SimpleGet(API):

    async def handle(self, http_body: Awaitable[bytes]):
        data = await http_body
        return json.loads(data.decode("utf8"))

    def collect(self, *responses):
        return responses[0]

    def _generate_urls(self):
        return ["https://httpbin.org/headers"]


class ApiTests(unittest.TestCase):

    def test_simple_get(self):
        http = SimpleGet()
        r = asyncio.run(http.run())
        self.assertEqual("httpbin.org", r[0]["headers"]["Host"])




if __name__ == '__main__':
    unittest.main()
