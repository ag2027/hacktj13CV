import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import { RoverProvider } from './context/RoverContext.jsx'
import './styles/tokens.css'
import './styles/global.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <RoverProvider>
      <App />
    </RoverProvider>
  </React.StrictMode>,
)