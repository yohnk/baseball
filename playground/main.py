import concurrent
import io
from threading import Thread

import aiohttp
import asyncio
import time
import pandas as pd
import aiofiles
import requests

from data.data_types import get_types
# from playground.TaskGraph import TaskGraph


class APISupplier:

    def __init__(self, game_id):
        self.game_id = game_id
        self.url = "https://baseballsavant.mlb.com/statcast_search/csv?all=true&game_pk={}&type=details".format(
            self.game_id)

    async def aioget(self):
        async with aiohttp.ClientSession() as session:
            async with session.get(self.url) as response:
                return self.game_id, await response.text()

    def get(self):
        return self.game_id, requests.get(self.url).content.decode('utf-8')


class PandasCreator:

    async def aiocreate(self, api_data_task):
        _, data = await api_data_task
        return pd.read_csv(io.StringIO(data))

    def create(self, api_data_task):
        _, data = api_data_task
        return pd.read_csv(io.StringIO(data))


class PandasCleaner:

    def __init__(self, types={}):
        self.types = types

    async def aioclean(self, data_frame):
        return (await data_frame).astype(self.types)

    def clean(self, data_frame):
        return data_frame.astype(self.types)


class DataCache:

    async def aiocache(self, api_data_task):
        cache_id, data = await api_data_task
        async with aiofiles.open('/tmp/{}.txt'.format(cache_id), mode='w') as f:
            await f.write(data)

    def cache(self, api_data_task):
        cache_id, data = api_data_task
        with open('/tmp/{}.txt'.format(cache_id), mode='w') as f:
            f.write(data)


async def main():
    game_ids = [661042, 661041, 661040, 661039, 661036, 661038]

    asp = {game_id: APISupplier(game_id) for game_id in game_ids}
    pc = PandasCreator()
    pcl = PandasCleaner(types=get_types()["statcast"])
    dc = DataCache()

    start = time.time()

    df_tasks = []
    cache_tasks = []
    for game_id in game_ids:
        api = asp[game_id]
        task1 = asyncio.create_task(api.aioget())
        task2 = asyncio.create_task(pc.aiocreate(task1))
        task3 = asyncio.create_task(pcl.aioclean(task2))
        df_tasks.append(task3)
        task4 = asyncio.create_task(dc.aiocache(task1))
        cache_tasks.append(task4)

    async_df = await asyncio.gather(*df_tasks)
    await asyncio.gather(*cache_tasks)

    print("Async Time", time.time() - start)

    start = time.time()

    sync_df = []

    for game_id in game_ids:
        api = asp[game_id]
        data = api.get()
        sync_df.append(pcl.clean(pc.create(data)))
        dc.cache(data)

    print("Sync Time", time.time() - start)

    graph = TaskGraph()
    graph.root().async_task(api.aioget).process_task(pc.create).split(
        graph.process_task(pcl.clean).collect(),
        graph.async_task(dc.aiocache)
    )

    result = graph.run()


async def test(output, f: concurrent.futures.Future):
    await asyncio.sleep(1)
    f.set_result(output)

if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    thread = Thread(target=loop.run_forever)
    thread.start()

    f1 = concurrent.futures.Future()
    _ = loop.call_soon_threadsafe(asyncio.ensure_future, test("One", f1))
    f2 = concurrent.futures.Future()
    _ = loop.call_soon_threadsafe(asyncio.ensure_future, test("Two", f2))

    fs = [f1, f2]

    while len(fs) > 0:
        done = len(fs) == 0 or all(fs)
        complete = [f for f in fs if f.done()]
        for f in complete:
            print(f.result())
        fs = [f for f in fs if f not in complete]

    loop.call_soon_threadsafe(loop.stop)
    while loop.is_running():
        print("Still running")
        thread.join(1)

    print("Joined")
    loop.close()


    # task = loop.create_task(test())


