# game_details = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_pk={}&type=details"
# game_dates = "https://statsapi.mlb.com/api/v1/schedule?startDate=01/01/{}&endDate=12/31/{}&sportId=1"
#
from collections.abc import Iterable

import aiohttp
from abc import ABC, abstractmethod
from typing import Awaitable
from aiohttp import ClientResponse, ClientSession, BaseConnector
from aiohttp_client_cache import CachedSession, FileBackend
import playground.tasks.TaskGraph as tg
from playground.tasks.TaskGraph import Node


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
            return await self.tree.start(raise_exception=True)
        finally:
            await self.session.close()

    @abstractmethod
    async def handle(self, http_body: Awaitable[bytes]):
        pass

    @abstractmethod
    async def collect(self, *responses):
        pass

    @staticmethod
    async def _check_request(response: ClientResponse) -> bool:
        return response is None

    def _generate_session(self) -> ClientSession:
        return ClientSession(connector=self._generate_connection())

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

    def __init__(self, cache_backend=None, **kwargs):
        self.cache_backend = cache_backend
        super().__init__(**kwargs)

    async def _async_init(self):
        self._init_cache()
        await super()._async_init()

    def _init_cache(self):
        if self.cache_backend is None:
            self.cache_backend = self._generate_cache()

    def _generate_cache(self):
        return FileBackend(cache_name=self._generate_cache_name(), urls_expire_after=self._generate_expiration())

    @staticmethod
    def _generate_expiration():
        return {}

    @staticmethod
    def _generate_cache_name():
        return "http_cache"

    def _generate_session(self):
        return CachedSession(cache=self.cache_backend, conn=self._generate_connection())


class RetryAPI(API, ABC):

    def __init__(self, max_retries=1):
        self.max_retries = max_retries
        self._retries = {}

    @staticmethod
    async def _retry_request(response):
        return response is None
