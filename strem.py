import asyncio
from telegram import Bot

# Initialize the bot with your token
bot_token = '6789002761:AAFi_JWa10dGoXIMDjv2AM_teaqtW6osgxM'  # Replace with your bot token
channel_id = '@telugumovieworld2'  # Replace with your channel ID
bot = Bot(token=bot_token)

async def send_video():
    # Replace 'looped_video.mp4' with the path to your video
    with open('output.mp4', 'rb') as video:
        await bot.send_video(chat_id=channel_id, video=video)

async def main():
    while True:
        await send_video()
        await asyncio.sleep(10)  # Adjust the sleep time based on your video's duration

if __name__ == "__main__":
    asyncio.run(main())
