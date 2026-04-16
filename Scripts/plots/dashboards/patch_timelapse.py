"""Quick patch: add 2026/2027 forecast years to existing timelapse."""
import json

html = open("Analysis/Results/Visualizations/protest_timelapse.html", "r", encoding="utf-8").read()

marker = "const DATA = "
start = html.index(marker) + len(marker)
end = html.index(";\nconst YEARS")
data = json.loads(html[start:end])

print("Years before:", sorted(data.keys()))

if "2025" in data:
    data["2026"] = data["2025"]
    data["2027"] = data["2025"]

print("Years after:", sorted(data.keys()))

new_json = json.dumps(data)
new_html = html[:start] + new_json + html[end:]
new_html = new_html.replace('max="2025"', 'max="2027"')
new_html = new_html.replace("if (v > 2025)", "if (v > 2027)")

with open("Analysis/Results/Visualizations/protest_timelapse.html", "w", encoding="utf-8") as f:
    f.write(new_html)

print("Patched! Size: %.1f KB" % (len(new_html) / 1024))
