const $ = (id) => document.getElementById(id);
const form = $('scenarioForm');
const API_URL = (window.FLIGHTRESCUE_API_URL || '').replace(/\/$/, '');
const apiBadge = $('apiBadge');
let lastResult = null;
let weatherTimer = null;

const AIRPORTS = [
  {code:'OGG',name:'Kahului Airport',city:'Kahului / Maui',state:'HI',lat:20.8987,lon:-156.4305},
  {code:'HNL',name:'Daniel K. Inouye International Airport',city:'Honolulu / Oahu',state:'HI',lat:21.3187,lon:-157.9224},
  {code:'KOA',name:'Ellison Onizuka Kona International Airport',city:'Kailua-Kona / Hawaii Island',state:'HI',lat:19.7388,lon:-156.0456},
  {code:'LIH',name:'Lihue Airport',city:'Lihue / Kauai',state:'HI',lat:21.9760,lon:-159.3390},
  {code:'ITO',name:'Hilo International Airport',city:'Hilo / Hawaii Island',state:'HI',lat:19.7203,lon:-155.0485},
  {code:'LAX',name:'Los Angeles International Airport',city:'Los Angeles',state:'CA',lat:33.9416,lon:-118.4085},
  {code:'SFO',name:'San Francisco International Airport',city:'San Francisco',state:'CA',lat:37.6213,lon:-122.3790},
  {code:'SJC',name:'Norman Y. Mineta San Jose International Airport',city:'San Jose',state:'CA',lat:37.3639,lon:-121.9289},
  {code:'OAK',name:'San Francisco Bay Oakland International Airport',city:'Oakland',state:'CA',lat:37.7126,lon:-122.2197},
  {code:'SAN',name:'San Diego International Airport',city:'San Diego',state:'CA',lat:32.7338,lon:-117.1933},
  {code:'SMF',name:'Sacramento International Airport',city:'Sacramento',state:'CA',lat:38.6954,lon:-121.5908},
  {code:'SEA',name:'Seattle-Tacoma International Airport',city:'Seattle',state:'WA',lat:47.4502,lon:-122.3088},
  {code:'PDX',name:'Portland International Airport',city:'Portland',state:'OR',lat:45.5898,lon:-122.5951},
  {code:'LAS',name:'Harry Reid International Airport',city:'Las Vegas',state:'NV',lat:36.0840,lon:-115.1537},
  {code:'PHX',name:'Phoenix Sky Harbor International Airport',city:'Phoenix',state:'AZ',lat:33.4342,lon:-112.0116},
  {code:'DEN',name:'Denver International Airport',city:'Denver',state:'CO',lat:39.8561,lon:-104.6737},
  {code:'SLC',name:'Salt Lake City International Airport',city:'Salt Lake City',state:'UT',lat:40.7899,lon:-111.9791},
  {code:'DFW',name:'Dallas Fort Worth International Airport',city:'Dallas / Fort Worth',state:'TX',lat:32.8998,lon:-97.0403},
  {code:'IAH',name:'George Bush Intercontinental Airport',city:'Houston',state:'TX',lat:29.9902,lon:-95.3368},
  {code:'ORD',name:"O'Hare International Airport",city:'Chicago',state:'IL',lat:41.9742,lon:-87.9073},
  {code:'MSP',name:'Minneapolis-Saint Paul International Airport',city:'Minneapolis / Saint Paul',state:'MN',lat:44.8848,lon:-93.2223},
  {code:'ATL',name:'Hartsfield-Jackson Atlanta International Airport',city:'Atlanta',state:'GA',lat:33.6407,lon:-84.4277},
  {code:'JFK',name:'John F. Kennedy International Airport',city:'New York',state:'NY',lat:40.6413,lon:-73.7781},
  {code:'EWR',name:'Newark Liberty International Airport',city:'Newark / New York',state:'NJ',lat:40.6895,lon:-74.1745},
  {code:'BOS',name:'Boston Logan International Airport',city:'Boston',state:'MA',lat:42.3656,lon:-71.0096},
  {code:'IAD',name:'Washington Dulles International Airport',city:'Washington',state:'VA',lat:38.9531,lon:-77.4565},
  {code:'MCO',name:'Orlando International Airport',city:'Orlando',state:'FL',lat:28.4312,lon:-81.3081},
  {code:'AUS',name:'Austin-Bergstrom International Airport',city:'Austin',state:'TX',lat:30.1975,lon:-97.6664},
  {code:'SAT',name:'San Antonio International Airport',city:'San Antonio',state:'TX',lat:29.5337,lon:-98.4698},
  {code:'YVR',name:'Vancouver International Airport',city:'Vancouver',state:'BC',lat:49.1967,lon:-123.1815},
  {code:'YYC',name:'Calgary International Airport',city:'Calgary',state:'AB',lat:51.1215,lon:-114.0076},
  {code:'ANC',name:'Ted Stevens Anchorage International Airport',city:'Anchorage',state:'AK',lat:61.1743,lon:-149.9985}
];

const airportDisplay = (a) => `${a.code} — ${a.name} — ${a.city}, ${a.state}`;
$('airportOptions').innerHTML = AIRPORTS.map(a => `<option value="${airportDisplay(a).replaceAll('"','&quot;')}"></option>`).join('');
$('fromAirport').value = airportDisplay(AIRPORTS.find(a=>a.code==='OGG'));
$('toAirport').value = airportDisplay(AIRPORTS.find(a=>a.code==='HNL'));

function setTheme(theme){document.documentElement.dataset.theme=theme;localStorage.setItem('flightrescue-theme',theme);document.querySelector('meta[name="theme-color"]').content=theme==='dark'?'#06111c':'#ffffff';}
setTheme(localStorage.getItem('flightrescue-theme') || (matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'));
$('themeToggle').addEventListener('click',()=>setTheme(document.documentElement.dataset.theme==='dark'?'light':'dark'));
function setApiState(text, cls=''){apiBadge.textContent=text;apiBadge.className=`status-badge ${cls}`;}

function airportFromInput(value){
  const raw=(value||'').trim();
  if(!raw)return null;
  const codeMatch=raw.toUpperCase().match(/^([A-Z0-9]{3})\b/);
  if(codeMatch){const byCode=AIRPORTS.find(a=>a.code===codeMatch[1]);if(byCode)return byCode;}
  const q=raw.toLowerCase();
  const exact=AIRPORTS.find(a=>[a.code,a.name,a.city,`${a.city}, ${a.state}`].some(x=>String(x).toLowerCase()===q));
  if(exact)return exact;
  const matches=AIRPORTS.filter(a=>`${a.code} ${a.name} ${a.city} ${a.state}`.toLowerCase().includes(q));
  return matches.length===1?matches[0]:null;
}

function haversineMiles(a,b){
  const R=3958.7613, rad=x=>x*Math.PI/180;
  const dLat=rad(b.lat-a.lat), dLon=rad(b.lon-a.lon);
  const s=Math.sin(dLat/2)**2+Math.cos(rad(a.lat))*Math.cos(rad(b.lat))*Math.sin(dLon/2)**2;
  return 2*R*Math.asin(Math.sqrt(s));
}

function routeInfo(showError=true){
  const from=airportFromInput($('fromAirport').value), to=airportFromInput($('toAirport').value), help=$('routeHelp');
  let error='';
  if(!from)error='Please choose the departure airport from the suggestions.';
  else if(!to)error='Please choose the destination airport from the suggestions.';
  else if(from.code===to.code)error='Departure and destination airports must be different.';
  else if(from.code!=='OGG' && to.code!=='OGG')error='This research model currently supports routes where Kahului Airport (OGG) is either the departure or arrival airport.';
  if(error){if(showError){help.textContent=error;help.className='help-text error';}return null;}
  const direction=from.code==='OGG'?'departure':'arrival';
  const other=direction==='departure'?to:from;
  help.textContent=`${direction==='departure'?'Departing from':'Arriving at'} Kahului Airport (OGG) · ${other.city} (${other.code}) · weather evaluated at OGG.`;
  help.className='help-text';
  $('dateLabel').textContent=`Scheduled ${direction==='departure'?'departure':'arrival'} date at OGG`;
  $('timeLabel').textContent=`Scheduled ${direction==='departure'?'departure':'arrival'} time at OGG`;
  return {from,to,direction,other,distance_miles:Math.round(haversineMiles(AIRPORTS[0],other))};
}

$('swapAirports').addEventListener('click',()=>{const a=$('fromAirport').value;$('fromAirport').value=$('toAirport').value;$('toAirport').value=a;routeInfo();scheduleWeatherPreview();});
['fromAirport','toAirport'].forEach(id=>$(id).addEventListener('change',()=>routeInfo()));

function hstDate(offset=0){
  const now=new Date(Date.now()+offset*86400000);
  const parts=new Intl.DateTimeFormat('en-US',{timeZone:'Pacific/Honolulu',year:'numeric',month:'2-digit',day:'2-digit'}).formatToParts(now);
  const get=t=>parts.find(p=>p.type===t).value;
  return `${get('year')}-${get('month')}-${get('day')}`;
}
function populateTimes(){
  const sel=$('flightTime');
  for(let m=0;m<24*60;m+=15){const h=Math.floor(m/60), min=m%60, ap=h<12?'AM':'PM', h12=(h%12)||12, value=`${String(h).padStart(2,'0')}:${String(min).padStart(2,'0')}`;const opt=document.createElement('option');opt.value=value;opt.textContent=`${h12}:${String(min).padStart(2,'0')} ${ap}`;sel.appendChild(opt);}
  sel.value='10:45';
}
populateTimes();
$('flightDate').value=hstDate(1);
$('flightDate').min=hstDate(0);
document.querySelectorAll('.quick-date').forEach(btn=>btn.addEventListener('click',()=>{$('flightDate').value=hstDate(Number(btn.dataset.offset));scheduleWeatherPreview();}));
function scheduledLocal(){return `${$('flightDate').value}T${$('flightTime').value}`;}

function weatherText(w){
  if(!w?.available)return w?.reason||'NWS forecast is not available for this requested time.';
  const bits=[w.short_forecast||'NWS hourly forecast'];
  if(w.temperature_f!=null)bits.push(`${Math.round(w.temperature_f)}°F`);
  if(w.humidity_pct!=null)bits.push(`${Math.round(w.humidity_pct)}% humidity`);
  if(w.wind_speed_mph!=null)bits.push(`wind ${Math.round(w.wind_speed_mph)} mph${w.wind_gust_mph!=null?` · gust ${Math.round(w.wind_gust_mph)} mph`:''}`);
  if(w.precipitation_probability_pct!=null)bits.push(`${Math.round(w.precipitation_probability_pct)}% precip chance`);
  return bits.join(' · ');
}

async function refreshWeatherPreview(){
  const box=$('weatherPreview');
  if(!API_URL || !$('flightDate').value || !$('flightTime').value)return;
  box.querySelector('p').textContent='Loading the official NWS forecast for Kahului Airport…';
  try{const res=await fetch(`${API_URL}/weather/ogg?scheduled_local=${encodeURIComponent(scheduledLocal())}`);const body=await res.json();if(!res.ok)throw new Error(body.detail||`HTTP ${res.status}`);box.querySelector('p').textContent=weatherText(body);}
  catch(err){box.querySelector('p').textContent=`NWS preview unavailable: ${err.message}. FlightRescue will retry during analysis.`;}
}
function scheduleWeatherPreview(){clearTimeout(weatherTimer);weatherTimer=setTimeout(refreshWeatherPreview,450);}
$('flightDate').addEventListener('change',scheduleWeatherPreview);$('flightTime').addEventListener('change',scheduleWeatherPreview);

function scenarioPayload(){
  const route=routeInfo(true);if(!route)throw new Error($('routeHelp').textContent);
  return {airline:$('airline').value,direction:route.direction,other_airport:route.other.code,scheduled_local:scheduledLocal(),distance_miles:route.distance_miles,event_type:$('eventType').value};
}
function setLoading(on){$('analyzeButton').disabled=on;$('analyzeButton').querySelector('span').textContent=on?'Analyzing…':'Analyze Risk';}
function showUnavailable(message){$('riskRing').textContent='—';$('riskLabel').textContent='AI service unavailable';$('resultMessage').textContent=message;$('severeLabel').textContent='—';$('recoveryLabel').textContent='—';$('confidenceLabel').textContent='Experimental';$('probabilityText').textContent='—';$('probabilityBar').style.width='0';}
function pct(v){return v==null?'—':`${(Number(v)*100).toFixed(1)}%`;}
function showWeather(r){const el=$('weatherSource'),w=r.weather||{},nws=w.nws||{};if(nws.available){const mode=w.preset_applied?' · hypothetical hazard stress test applied':'';el.textContent=`NWS at OGG: ${weatherText(nws)} · detected category: ${String(w.effective_event_type||'normal').replaceAll('_',' ')}${mode}`;}else{el.textContent=`NWS forecast unavailable for this requested time. ${w.preset_applied?'The selected hypothetical hazard preset was used for weather stress testing.':'Missing weather fields were handled by the trained model pipeline.'}`;}}

function renderComparison(comp){
  const box=$('airlineComparison');
  if(!comp?.available){box.innerHTML='<div class="empty-state">Airline-by-event BTS comparison is not available yet for these analogs. If the new aggregate is still building, this panel will activate automatically after deployment.</div>';return;}
  const rows=comp.rows||[];
  if(!rows.length){box.innerHTML='<div class="empty-state">No airline flight observations were available in the matched historical event windows.</div>';return;}
  const body=rows.map(r=>`<tr class="${r.airline===comp.selected_airline?'selected-airline':''}"><td>${r.airline_name} <small>(${r.airline})</small></td><td>${Math.round(r.total_flights)}</td><td>${pct(r.delay_rate)}</td><td>${pct(r.cancellation_rate)}</td><td>${pct(r.severe_rate)}</td><td>${r.mean_delay_minutes==null?'—':`${Number(r.mean_delay_minutes).toFixed(0)} min`}</td></tr>`).join('');
  box.innerHTML=`<h3>Airline performance across ${comp.analogs_used} matched historical event${comp.analogs_used===1?'':'s'}</h3><p>Selected airline is highlighted. Percentages are weighted from BTS OGG flight outcomes during the matched NOAA event windows.</p><table class="comparison-table"><thead><tr><th>Airline</th><th>Flights</th><th>Delayed ≥15m</th><th>Canceled</th><th>Severe</th><th>Mean delay</th></tr></thead><tbody>${body}</tbody></table>`;
}

function showResult(r){
  lastResult=r;
  const p=Math.max(0,Math.min(1,Number(r.disruption_probability||0))),pctRisk=Math.round(p*100),severe=Math.round(Math.max(0,Math.min(p,Number(r.severe_disruption_probability||0)))*100);
  $('riskRing').textContent=`${pctRisk}%`;$('riskLabel').textContent=`${String(r.risk_level||'model').replaceAll('_',' ')} disruption risk`;$('resultMessage').textContent='Trained OGG model using official NWS weather when available, with historical airline evidence shown separately.';$('severeLabel').textContent=`${severe}%`;
  const recovery=r.recovery?.median_hours;$('recoveryLabel').textContent=recovery!=null?`~${Math.round(recovery)} h`:'No storm analog';$('confidenceLabel').textContent=r.confidence||'Experimental';$('probabilityText').textContent=`${pctRisk}%`;$('probabilityBar').style.width=`${pctRisk}%`;
  const c=pctRisk>=75?'var(--danger)':pctRisk>=50?'var(--warn)':pctRisk>=30?'var(--accent)':'var(--success)';$('riskRing').style.borderColor=c;showWeather(r);
  const n=r.airline_comparison?.analogs_used||0;$('analogSummary').textContent=n?`${n} historical hazardous-weather event${n===1?'':'s'} matched the current NWS category. See the airline comparison below for delay and cancellation behavior.`:'No comparable hazardous-weather event category was needed/found for this forecast. The ML risk still uses the available flight and NWS weather inputs.';
  renderComparison(r.airline_comparison);renderEvents(r.historical_context||[]);
}

function percentValue(v){if(v==null)return null;const n=Number(v);return Math.abs(n)<=1?n*100:n;}
function eventCard(e){const title=e.event_types||e.event_type||e.EVENT_TYPE||e.event_id||'Historical weather episode',date=e.start_dt||e.start_date||'',recovery=e.recovery_hours_after_event??e.recovery_hours,cancel=percentValue(e.event_cancel_rate);return `<article class="event-card"><h3>${String(title)}</h3>${date?`<p>${String(date).slice(0,10)}</p>`:''}${cancel!=null?`<p>Overall OGG cancellation rate: <strong>${cancel.toFixed(1)}%</strong></p>`:''}${recovery!=null?`<p>Recovery proxy: <strong>${Number(recovery).toFixed(1)} h</strong></p>`:''}</article>`;}
function renderEvents(events){const box=$('historicalEvents');box.innerHTML=events.length?events.map(eventCard).join(''):'<div class="empty-state">No hazardous-weather analog episodes were returned for this forecast category.</div>';}

form.addEventListener('submit',async(e)=>{e.preventDefault();if(!API_URL){showUnavailable('The public backend URL has not been configured yet.');return;}let payload;try{payload=scenarioPayload();}catch(err){showUnavailable(err.message);return;}setLoading(true);try{const res=await fetch(`${API_URL}/predict/scenario`,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const body=await res.json();if(!res.ok)throw new Error(body.detail||`HTTP ${res.status}`);showResult(body);}catch(err){showUnavailable(err.message||'Prediction request failed.');}finally{setLoading(false);}});

async function loadHistory(){if(lastResult){renderComparison(lastResult.airline_comparison);renderEvents(lastResult.historical_context||[]);return;}const type=$('eventType').value;if(type==='auto'){renderEvents([]);return;}try{const res=await fetch(`${API_URL}/historical/context?event_type=${encodeURIComponent(type)}&k=6`);const body=await res.json();if(!res.ok)throw new Error(body.detail||`HTTP ${res.status}`);renderEvents(body.events||[]);}catch(err){$('historicalEvents').innerHTML=`<div class="empty-state">Could not load historical context: ${err.message}</div>`;}}
$('historyButton').addEventListener('click',loadHistory);

async function checkApi(){if(!API_URL){setApiState('Backend pending','offline');showUnavailable('Frontend deployed. Waiting for the public model API URL.');return;}try{const res=await fetch(`${API_URL}/health`);const s=await res.json();if(res.ok&&s.ready){setApiState(s.airline_event_performance?'AI + NWS + BTS online':'AI + NWS online','online');$('riskLabel').textContent='AI model online';$('resultMessage').textContent='Choose a flight. FlightRescue will retrieve official NWS weather automatically.';}else{setApiState('Artifacts loading','offline');showUnavailable('API is online, but trained model artifacts are not loaded yet.');}}catch(_){setApiState('API offline','offline');showUnavailable('The model API could not be reached.');}}

routeInfo(false);checkApi();scheduleWeatherPreview();
