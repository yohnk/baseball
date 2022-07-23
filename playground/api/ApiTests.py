import asyncio
import math
import time
import unittest
from email.utils import format_datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from typing import Awaitable
from aiohttp import ClientResponseError
from datetime import datetime, timedelta
from playground.api.API import API, RetryAPI


class WebServer(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes("Hello World", "utf8"))
        elif "/status/" in self.path:
            self.send_response_only(int(self.path.replace("/status/", "")))
            self.end_headers()
        elif "/retry/" in self.path:
            self.send_response(301)
            self.send_header("Retry-After", self.path.replace("/retry/", ""))
            self.end_headers()
        elif "/retry_date/" in self.path:
            seconds = int(self.path.replace("/retry_date/", ""))
            self.send_response(429)
            self.send_header("Retry-After", format_datetime(datetime.now() + timedelta(seconds=seconds)))
            self.end_headers()
        else:
            self.send_response_only(400)
            self.end_headers()


class SimpleGet(API):

    async def handle(self, http_body: Awaitable[bytes]):
        data = await http_body
        return data.decode("utf8")

    def collect(self, *responses):
        if len(responses) == 1:
            return responses[0]
        else:
            return responses

    def _generate_urls(self):
        return ["http://localhost:8080"]


class RetryGet(RetryAPI):

    async def handle(self, http_body: Awaitable[bytes]):
        data = await http_body
        return data.decode("utf8")

    def collect(self, *responses):
        if len(responses) == 1:
            return responses[0]
        else:
            return responses

    def _generate_urls(self):
        return ["http://localhost:8080"]


class ApiTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.httpd = HTTPServer(('localhost', 8080), WebServer)
        cls.http_thread = Thread(target=cls.httpd.serve_forever)
        cls.http_thread.start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.httpd.shutdown()
        cls.http_thread.join()

    def test_simple_get(self):
        http = SimpleGet()
        r = asyncio.run(http.run())
        self.assertEqual("Hello World", r[0])

    def test_500_get(self):
        http = SimpleGet(urls=["http://localhost:8080/status/500"])
        asyncio.run(http.run())
        self.assertEqual(1, len(http.tree.exceptions))
        self.assertEqual(type(http.tree.exceptions[0]), ClientResponseError)
        self.assertEqual(500, http.tree.exceptions[0].status)

    def test_retry_simple_get(self):
        http = RetryGet()
        r = asyncio.run(http.run())
        self.assertEqual("Hello World", r[0])

    def test_retry_500(self):
        url = "http://localhost:8080/status/500"
        http = RetryGet(urls=[url], max_retries=2)
        r = asyncio.run(http.run())
        self.assertEqual(1, len(http.tree.exceptions))
        self.assertEqual(type(http.tree.exceptions[0]), ClientResponseError)
        self.assertEqual(500, http.tree.exceptions[0].status)
        self.assertEqual(1, len(http._retries))
        self.assertEqual(2, list(http._retries.values())[0])
        self.assertEqual("http://localhost:8080/status/500", str(list(http._retries.keys())[0]))

    def test_retry_time(self):
        http = RetryGet(urls=["http://localhost:8080/status/500"], max_retries=3)
        start = time.time()
        asyncio.run(http.run())
        total = time.time() - start
        print(total)
        # Timing methods is gross, so shoot for a window
        # Default is exponential decay
        self.assertTrue(math.pow(2, 3) > total >= math.pow(2, 2))
        self.assertEqual(500, http.tree.exceptions[0].status)

    def test_retry_after_seconds(self):
        http = RetryGet(urls=["http://localhost:8080/retry/3"], max_retries=3)
        start = time.time()
        asyncio.run(http.run())
        total = time.time() - start
        print(total)
        # Timing methods is gross, so shoot for a window
        self.assertTrue(7 > total >= 5)
        self.assertEqual(301, http.tree.exceptions[0].status)

    def test_retry_after_date(self):
        http = RetryGet(urls=["http://localhost:8080/retry_date/3", "http://localhost:8080/retry_date/2"], max_retries=3)
        start = time.time()
        asyncio.run(http.run())
        total = time.time() - start
        print(total)
        # Timing methods is gross, so shoot for a window
        self.assertTrue(7 > total >= 5)
        self.assertEqual(429, http.tree.exceptions[0].status)


if __name__ == '__main__':
    unittest.main()
