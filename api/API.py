# game_details = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_pk={}&type=details"
# game_dates = "https://statsapi.mlb.com/api/v1/schedule?startDate=01/01/{}&endDate=12/31/{}&sportId=1"
#
import asyncio
import math
from collections.abc import Iterable
from datetime import datetime
import aiohttp
from abc import ABC, abstractmethod
from typing import Awaitable
from aiohttp import ClientResponse, ClientSession, BaseConnector, ClientResponseError
from aiohttp_client_cache import CachedSession, FileBackend, CacheBackend
import TaskGraph as tg
from TaskGraph import Node
from email.utils import parsedate_to_datetime


class API(ABC):

    def __init__(self, session: ClientSession = None, urls: Iterable[str] = None, tree: Node = None, conn_limit: int = 5):
        self.session = session
        self.urls = urls
        self.tree = tree
        self.result = None
        self.conn_limit = conn_limit

    async def _async_init(self):
        await self._init_http()
        self._init_work()

    async def _init_http(self):
        if self.session is None:
            self.session = self._generate_session()
        if self.urls is None:
            self.urls = self._generate_urls()

    def _init_work(self):
        if self.tree is None:
            self.tree = self._generate_tree()

    async def _http(self, url):
        response = None
        while await self._check_request(response):
            async with self.session.get(await url) as response:
                body = await response.read()
        return body

    async def run(self):
        await self._async_init()
        try:
            return await self.tree.start()
        finally:
            await self.session.close()

    @abstractmethod
    async def handle(self, http_body: Awaitable[bytes]):
        pass

    @abstractmethod
    async def collect(self, *responses):
        pass

    async def _check_request(self, response: ClientResponse) -> bool:
        return response is None

    def _generate_session(self) -> ClientSession:
        return ClientSession(connector=self._generate_connection(), raise_for_status=True)

    def _generate_connection(self) -> BaseConnector:
        return aiohttp.TCPConnector(limit=self.conn_limit)

    def _generate_urls(self) -> Iterable[str]:
        return []

    def _generate_tree(self) -> Node:
        collect = tg.MainNode(self.collect)
        for url in self.urls:
            tg.SeedNode(url).async_task(self._http).async_task(self.handle).add_child(collect)
        return collect


class CachedAPI(API, ABC):

    def __init__(self, cache_backend: CacheBackend = None, **kwargs):
        self.cache_backend: CacheBackend = cache_backend
        super().__init__(**kwargs)

    async def _async_init(self):
        self._init_cache()
        await super()._async_init()

    def _init_cache(self):
        if self.cache_backend is None:
            self.cache_backend = self._generate_cache()

    def _generate_cache(self):
        return FileBackend(cache_name=self._generate_cache_name(), urls_expire_after=self._generate_expiration())

    def _generate_expiration(self):
        return {}

    def _generate_cache_name(self):
        return "http_cache"

    def _generate_session(self):
        return CachedSession(cache=self.cache_backend, conn=self._generate_connection())


class RetryAPI(API, ABC):

    def __init__(self, max_retries=1, **kwargs):
        self.max_retries = max_retries
        self._retries = {}
        super().__init__(**kwargs)

    async def _check_request(self, response: ClientResponse) -> bool:
        if response is None:
            return True
        if response.status >= 500 or response.status == 429 or response.status == 301:
            if response.url not in self._retries:
                self._retries[response.url] = 0

            self._retries[response.url] += 1

            if self._retries[response.url] >= self.max_retries:
                raise ClientResponseError(request_info=response.request_info, status=response.status, message=response.reason, headers=response.headers, history=(response,))

            if "Retry-After" in response.headers and self.get_int(response.headers["Retry-After"]) is not None:
                await asyncio.sleep(self.get_int(response.headers["Retry-After"]))
            elif "Retry-After" in response.headers and self.get_date(response.headers["Retry-After"]) is not None:
                wake = self.get_date(response.headers["Retry-After"])
                while datetime.now() < wake:
                    td = wake - datetime.now()
                    await asyncio.sleep(td.total_seconds() * 0.99)
            else:
                await asyncio.sleep(math.pow(2, self._retries[response.url]))

            return True
        else:
            return False


    @staticmethod
    def get_int(o):
        if isinstance(o, int):
            return o
        if isinstance(o, str) and o.isdigit():
            return int(o)
        return None

    @staticmethod
    def get_date(o):
        try:
            return parsedate_to_datetime(o)
        except Exception:
            return None

    def _generate_session(self) -> ClientSession:
        return ClientSession(connector=self._generate_connection(), raise_for_status=False)

