import asyncio

import aiohttp


async def http(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            print(response.status)


async def main():
    f = await http("https://httpbin.org/ip")
    # f = await f
    # f = await f
    print(f)


if __name__ == '__main__':
    asyncio.run(main())