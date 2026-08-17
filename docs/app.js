const form = document.getElementById('scenarioForm');
const risk = document.getElementById('riskRing');
const riskLabel = document.getElementById('riskLabel');
const severeLabel = document.getElementById('severeLabel');
const recoveryLabel = document.getElementById('recoveryLabel');
const confidenceLabel = document.getElementById('confidenceLabel');

const API_URL = (window.FLIGHTRESCUE_API_URL || '').replace(/\/$/, '');

function showNotConnected() {
  risk.textContent = '—';
  riskLabel.textContent = 'Live model API not connected yet';
  severeLabel.textContent = 'Unavailable';
  recoveryLabel.textContent = 'Historical engine ready';
  confidenceLabel.textContent = 'Research demo';
  risk.style.borderColor = '#6b7280';
}

function showApiResult(result) {
  const pct = Math.round((result.disruption_probability || 0) * 100);
  risk.textContent = `${pct}%`;
  riskLabel.textContent = `${String(result.risk_level || 'unknown').replace('_', ' ')} disruption risk`;
  severeLabel.textContent = `${Math.round((result.severe_disruption_probability || 0) * 100)}%`;
  recoveryLabel.textContent = result.estimated_recovery_hours != null
    ? `~${Math.round(result.estimated_recovery_hours)} h`
    : 'Historical context only';
  confidenceLabel.textContent = result.confidence || 'Model output';
  risk.style.borderColor = pct >= 75 ? '#fb7185' : pct >= 55 ? '#f59e0b' : pct >= 35 ? '#6ee7ff' : '#1f5f79';
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  if (!API_URL) {
    showNotConnected();
    return;
  }

  // The public form is currently a product mockup. It does not yet expose the
  // full leakage-safe feature dictionary expected by /predict/features.
  // Keeping this explicit prevents the UI from fabricating model probabilities.
  risk.textContent = '…';
  riskLabel.textContent = 'API connected — feature adapter pending';
  severeLabel.textContent = '—';
  recoveryLabel.textContent = '—';
  confidenceLabel.textContent = 'Backend available';
});

async function checkApi() {
  if (!API_URL) {
    showNotConnected();
    return;
  }
  try {
    const response = await fetch(`${API_URL}/health`);
    const status = await response.json();
    if (status.ready) {
      riskLabel.textContent = 'Model API online';
      confidenceLabel.textContent = 'Artifacts loaded';
    } else {
      riskLabel.textContent = 'API online — model artifacts not loaded';
      confidenceLabel.textContent = 'Setup required';
    }
  } catch (err) {
    riskLabel.textContent = 'Model API unreachable';
    confidenceLabel.textContent = 'Check backend';
  }
}

checkApi();
