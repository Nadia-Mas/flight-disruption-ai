// Production FastAPI base URL for FlightRescue AI.
// Passenger UI release: airport search + HST scheduling + NWS weather + airline comparison.
// Clean Pages deployment trigger after GitHub service/artifact retry issues.
window.FLIGHTRESCUE_API_URL = 'https://flight-disruption-ai.vercel.app';

// Lightweight traveler-persona animation module. Kept separate from inference logic.
(() => {
  const script = document.createElement('script');
  script.src = 'travelers.js';
  script.defer = true;
  document.head.appendChild(script);
})();
