SENTINEL Website (prototype)

Files:
- index.html — homepage
- livefeed.html — live camera placeholder and detections list
- map.html — tactical map placeholder
- styles.css, app.js — front-end assets

Quick start (local):

Open `website/index.html` in a browser for a static preview. For features that require WebSocket or fetch from `http` origins, serve the folder with a local server:

Windows / Python 3:

```powershell
cd website
python -m http.server 8000
```

Then open http://localhost:8000 in your browser.

Next steps:
- Integrate WebSocket or WebRTC live stream from `cv_server.py` or rover backends.
- Add mapping via Leaflet or Mapbox and pin detections with geolocation.
- Implement auth and secure command-room integration (Ro.am) and marketplace links.
