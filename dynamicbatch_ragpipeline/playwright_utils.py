from dynamicbatch_ragpipeline.env import args
from playwright.async_api import async_playwright
from datetime import datetime
from fastapi import HTTPException
from io import BytesIO
import time
import logging
import asyncio

playwrights = {}

async def initialize_browser(index, clear_first = False):
    global playwrights

    if clear_first:
        logging.info(f'clearing playwright {index}')
        for k in list(playwrights[index]):
            try:
                await playwrights[index][k].close()
            except:
                pass
            try:
                del playwrights[index][k]
            except:
                pass

    logging.info(f'initializing playwright {index}')

    playwright = await async_playwright().start()
    browser = await playwright.chromium.launch(headless = True)
    page = await browser.new_page()
    playwrights[index] = {
        'playwright': playwright,
        'browser': browser,
        'page': page,
        'available': True,
        'last_emit': datetime.now()
    }

async def dead(index):
    
    died = False
    try:
        if playwrights[index]['page'].is_closed():
            died = True
    except:
        pass

    try:
        if not playwrights[index]['browser'].is_connected():
            died = True
    except:
        pass

    if died:
        await initialize_browser(index=index, clear_first=True)

    return died

async def to_pdf(url, viewport_weight, viewport_height):
    index = 0
    found = False
    try:
        while True:
            for index in range(args.playwright_max_concurrency):
                if playwrights[index]['available']:
                    playwrights[index]['available'] = False
                    found = True
                    break

                await asyncio.sleep(1e-9)

            if found:
                break
        
        await playwrights[index]['page'].set_viewport_size({"width": viewport_weight, "height": viewport_height})
        await playwrights[index]['page'].goto(url)
        pdf = await playwrights[index]['page'].pdf()
        playwrights[index]['available'] = True
        return BytesIO(pdf)
        
    except Exception as e:
        await dead(index)
        playwrights[index]['available'] = True
        raise HTTPException(status_code=500, detail=f'failed, {e}, please retry.')

    
    

