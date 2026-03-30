import re
import json

def parse_swagit_playlist(html_text):
    # jwplayer("player").setup({ ... playlist: [{"id":1241656,"seq":2,"title":"Call to Order",...}] ... });
    match = re.search(r'playlist:\s*(\[\{.*?\}\])\s*,', html_text, re.DOTALL | re.IGNORECASE)
    if match:
        playlist_str = match.group(1)
        try:
            playlist = json.loads(playlist_str)
            return playlist
        except Exception as e:
            print(f"Error parsing JSON: {e}")
    else:
        # Sometimes different format
        match2 = re.search(r'playlist:\s*(\[\{.*?\}\])\s*\}', html_text, re.DOTALL | re.IGNORECASE)
        if match2:
            return json.loads(match2.group(1))
    return None

if __name__ == "__main__":
    import requests
    url = "http://austintx.swagit.com/play/45757"
    r = requests.get(url)
    pl = parse_swagit_playlist(r.text)
    if pl:
        print(f"Found {len(pl)} items in playlist.")
        for item in pl[:5]:
            print(f"Item: {item.get('title')} -> {item.get('dfile')}")
    else:
        print("Failed to find playlist.")
