// Minimal JS scaffold for later integration with rover backends

document.addEventListener('DOMContentLoaded',()=>{
  // Example: show a banner if we're running under file:// vs http(s)
  if(location.protocol === 'file:'){
    console.info('Serving from file:// — for live features use a local server (see README).')
  }
});

// Placeholder functions to be replaced with real websocket/REST integrations.
function connectLiveFeed(wsUrl){
  // Return a dummy object; real implementation will use WebSocket/MediaStreams
  console.log('connectLiveFeed called', wsUrl);
  return {connected:false};
}

function addDetectionPin(detection){
  // detection: {id, lat, lng, label, severity, ts}
  const list = document.getElementById('pins-list')
  if(!list) return
  const li = document.createElement('li')
  li.textContent = `${new Date(detection.ts).toLocaleString()} — ${detection.label} [${detection.severity}]`
  list.prepend(li)
}

window.connectLiveFeed = connectLiveFeed
window.addDetectionPin = addDetectionPin
