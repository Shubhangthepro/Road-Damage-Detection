let map, markersLayer;

function priorityColor(priority) {
  if (priority === 'High') return '#ef4444';
  if (priority === 'Medium') return '#f59e0b';
  return '#22c55e';
}

function initMap() {
  map = L.map('map').setView([20.5937, 78.9629], 5); // India centroid default
  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap'
  }).addTo(map);
  markersLayer = L.layerGroup().addTo(map);
}

function markerIcon(color) {
  const svg = encodeURIComponent(
    `<svg xmlns='http://www.w3.org/2000/svg' width='30' height='30'>` +
    `<circle cx='15' cy='15' r='10' fill='${color}' stroke='white' stroke-width='2'/>` +
    `</svg>`
  );
  return L.icon({
    iconUrl: `data:image/svg+xml,${svg}`,
    iconSize: [30, 30],
    iconAnchor: [15, 15]
  });
}

function renderReports(reports) {
  // Map markers
  markersLayer.clearLayers();
  const bounds = [];
  reports.forEach(r => {
    const lat = r.location?.latitude;
    const lng = r.location?.longitude;
    if (lat != null && lng != null) {
      const color = priorityColor(r.priority);
      const m = L.marker([lat, lng], { icon: markerIcon(color) });
      const html = `<div class='space-y-1'>
        <div class='font-medium'>${r.category || 'RoadDamage'} — <span style='color:${color}'>${r.priority}</span></div>
        <div class='text-sm text-gray-600'>${r.location?.address || ''}</div>
        <div class='text-xs'>Severity: ${r.severity?.level || ''}, Risk: ${(r.predictiveRisk ?? 0).toFixed(2)}, Density: ${r.reportDensity ?? 0}</div>
      </div>`;
      m.bindPopup(html);
      m.addTo(markersLayer);
      bounds.push([lat, lng]);
    }
  });
  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30] });

  // List
  const list = document.getElementById('report-list');
  list.innerHTML = '';
  reports.forEach(r => {
    const color = priorityColor(r.priority);
    const card = document.createElement('div');
    card.className = 'border rounded p-3';
    card.innerHTML = `
      <div class='flex gap-3'>
        <img src='${r.imageUrl || ''}' class='w-20 h-20 object-cover rounded border' onerror="this.style.display='none'" />
        <div class='flex-1 min-w-0'>
          <div class='flex items-center justify-between'>
            <div class='font-medium truncate'>${r.location?.address || 'Unknown'}</div>
            <span class='text-xs px-2 py-0.5 rounded' style='background:${color}20;color:${color};border:1px solid ${color}'>${r.priority}</span>
          </div>
          <div class='text-xs text-gray-600 mt-1'>${new Date(r.createdAt).toLocaleString()}</div>
          <div class='text-xs mt-1'>Severity: ${r.severity?.level || ''} • Status: ${r.status}</div>
          <div class='text-xs text-gray-600'>Risk: ${(r.predictiveRisk ?? 0).toFixed(2)} • Density: ${r.reportDensity ?? 0}</div>
          <div class='text-xs text-gray-500 mt-1'>Reporter: ${r.reporter?.name || 'Anonymous'}${r.reporter?.email ? ` (${r.reporter.email})` : ''}</div>
          <div class='mt-2 flex gap-2'>
            <select class='border rounded px-2 py-1 text-sm' data-id='${r.id}' data-action='status'>
              ${['New','Scheduled','In Progress','Resolved'].map(s=>`<option ${s===r.status?'selected':''}>${s}</option>`).join('')}
            </select>
            <input class='border rounded px-2 py-1 text-sm' placeholder='Assign unit' data-id='${r.id}' data-field='unit'/>
            <button class='bg-gray-800 text-white rounded px-2 py-1 text-sm' data-id='${r.id}' data-action='assign'>Assign</button>
          </div>
        </div>
      </div>`;
    list.appendChild(card);
  });

  // Wire actions
  list.querySelectorAll('select[data-action="status"]').forEach(el => {
    el.addEventListener('change', async (e) => {
      const id = e.target.getAttribute('data-id');
      const status = e.target.value;
      await fetch(`/api/reports/${id}/status`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ status }) });
      loadReports();
    });
  });
  list.querySelectorAll('button[data-action="assign"]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const id = btn.getAttribute('data-id');
      const unit = btn.parentElement.querySelector('input[data-field="unit"]').value.trim();
      if (!unit) return;
      await fetch(`/api/reports/${id}/assign`, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({ unit }) });
      loadReports();
    });
  });
}

async function loadReports() {
  const severity = document.getElementById('filter-severity').value;
  const status = document.getElementById('filter-status').value;
  const sort = document.getElementById('filter-sort').value;
  const qs = new URLSearchParams();
  if (severity) qs.set('severity', severity);
  if (status) qs.set('status', status);
  if (sort) qs.set('sort', sort);
  const res = await fetch(`/api/reports?${qs.toString()}`);
  const json = await res.json();
  if (json.success) renderReports(json.reports);
}

window.addEventListener('DOMContentLoaded', () => {
  initMap();
  document.getElementById('refresh').addEventListener('click', loadReports);
  loadReports();
  // simple polling
  setInterval(loadReports, 15000);
});


