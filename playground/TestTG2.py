import asyncio
import io
import time
from concurrent.futures import ThreadPoolExecutor, Future

import pandas as pd

from TaskGraph2 import AsyncNode, SeedNode, CollectNode, MainNode, is_async
import numpy as np
import aiohttp


async def async_get_num(id):
    return id


async def do_http(num):
    if asyncio.iscoroutine(num):
        real_num = await num
    else:
        real_num = num

    async with aiohttp.ClientSession() as session:
        async with session.get("http://localhost:8000/data.csv?param={}".format(real_num)) as response:
            return await response.text()


async def parse_response(data):
    return pd.read_csv(io.StringIO(await data))


def std_all(*args):
    df = pd.concat(args)
    return df["release_pos_x"].std()


async def async_main():
    final_tasks = []
    game_ids = [661042, 661041, 661040, 661039, 661036, 661038]
    for id in game_ids:
        task1 = asyncio.create_task(async_get_num(id))
        task2 = asyncio.create_task(do_http(task1))
        final_tasks.append(asyncio.create_task(parse_response(task2)))
    return std_all(*[await f for f in final_tasks])


async def graph_main():
    root = AsyncNode()
    std = MainNode(work=std_all)
    # collector = CollectNode()
    # game_ids = [661042]
    game_ids = [661042, 661041, 661040, 661039, 661036, 661038]
    for id in game_ids:
        root.add_child(SeedNode(value=id)).add_child(AsyncNode(work=do_http)).add_child(AsyncNode(work=parse_response)).add_child(std)
    await root.start()
    return std.result()


if __name__ == "__main__":
    start = time.time()
    out1 = asyncio.run(async_main())
    print("Asyncio", time.time() - start, out1)

    start = time.time()
    out2 = asyncio.run(graph_main())
    print("Graph", time.time() - start, out2)

