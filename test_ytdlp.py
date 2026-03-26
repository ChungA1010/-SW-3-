import yt_dlp

url = "https://youtu.be/Z5sx7Zj5gKE?si=L5qplvbrQ5pnSuTT"

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