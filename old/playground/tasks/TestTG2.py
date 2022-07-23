import asyncio
import io
import time
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

from TaskGraph import SeedNode, ProcessNode
import aiohttp

e = ProcessPoolExecutor()


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
    out = df["release_pos_x"].std()
    return out


async def async_main():
    final_tasks = []
    game_ids = [661042, 661041, 661040, 661039, 661036, 661038]
    for id in game_ids:
        task1 = asyncio.create_task(async_get_num(id))
        task2 = asyncio.create_task(do_http(task1))
        final_tasks.append(asyncio.create_task(parse_response(task2)))
    f = e.submit(std_all, *[await f for f in final_tasks])
    while not f.done():
        pass
    return f.result()


async def graph_main():
    std = ProcessNode(std_all, executor=e)
    for id in [661042, 661041, 661040, 661039, 661036, 661038]:
        SeedNode(id).async_task(do_http).async_task(parse_response).add_child(std)
    out = await std.collect().start()
    return out


if __name__ == "__main__":
    start = time.time()
    out1 = asyncio.run(async_main())
    print("Asyncio", time.time() - start, out1)

    start = time.time()
    out2 = asyncio.run(graph_main())
    print("Graph", time.time() - start, out2)

    e.shutdown()

