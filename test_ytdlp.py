import yt_dlp

url = "https://youtu.be/7Bnir0BL1S4?si=8zv2atZ5Dv20iaws"

ydl_opts = {
    'format': 'bestaudio/best',
    'outtmpl': 'test_separation/_original.%(ext)s',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
    }],
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])