let map, markersLayer;

function statusColor(status) {
  const colors = {
    'New': '#3b82f6',
    'Scheduled': '#8b5cf6',
    'In Progress': '#f59e0b',
    'Resolved': '#22c55e'
  };
  return colors[status] || '#6b7280';
}

function priorityColor(priority) {
  if (priority === 'High') return '#ef4444';
  if (priority === 'Medium') return '#f59e0b';
  return '#22c55e';
}

function initMap() {
  map = L.map('map').setView([20.5937, 78.9629], 5);
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

async function loadMilestones(reportId) {
  try {
    const res = await fetch(`/api/user/milestones/${reportId}`);
    const json = await res.json();
    if (json.success) {
      return json.milestones || [];
    }
  } catch (e) {
    console.error('Failed to load milestones', e);
  }
  return [];
}

function renderReports(reports) {
  const list = document.getElementById('report-list');
  list.innerHTML = '';
  
  if (reports.length === 0) {
    list.innerHTML = '<div class="text-center text-gray-500 py-8">No reports yet. <a href="/" class="text-blue-600">Submit your first report</a></div>';
    return;
  }

  // Update stats
  const stats = {
    total: reports.length,
    resolved: reports.filter(r => r.status === 'Resolved').length,
    inProgress: reports.filter(r => r.status === 'In Progress').length,
    new: reports.filter(r => r.status === 'New').length
  };
  document.getElementById('total-reports').textContent = stats.total;
  document.getElementById('resolved-reports').textContent = stats.resolved;
  document.getElementById('in-progress-reports').textContent = stats.inProgress;
  document.getElementById('new-reports').textContent = stats.new;

  // Map markers
  markersLayer.clearLayers();
  const bounds = [];
  
  reports.forEach(r => {
    const lat = r.location?.latitude;
    const lng = r.location?.longitude;
    const color = statusColor(r.status);
    
    if (lat != null && lng != null) {
      const m = L.marker([lat, lng], { icon: markerIcon(color) });
      m.bindPopup(`<div class='font-medium'>${r.location?.address || 'Unknown'}</div><div class='text-sm'>Status: ${r.status}</div>`);
      m.addTo(markersLayer);
      bounds.push([lat, lng]);
    }

    // Report card
    const card = document.createElement('div');
    card.className = 'border rounded p-4 hover:shadow-md transition';
    const statusBg = statusColor(r.status);
    
    card.innerHTML = `
      <div class='flex gap-3'>
        <img src='${r.imageUrl || ''}' class='w-24 h-24 object-cover rounded border' onerror="this.style.display='none'" />
        <div class='flex-1 min-w-0'>
          <div class='flex items-center justify-between mb-2'>
            <div class='font-medium truncate'>${r.location?.address || 'Unknown'}</div>
            <span class='text-xs px-2 py-1 rounded text-white' style='background:${statusBg}'>${r.status}</span>
          </div>
          <div class='text-xs text-gray-600 mb-2'>${new Date(r.createdAt).toLocaleString()}</div>
          <div class='text-xs mb-2'>
            <span class='px-2 py-0.5 rounded' style='background:${priorityColor(r.priority)}20;color:${priorityColor(r.priority)};border:1px solid ${priorityColor(r.priority)}'>${r.priority} Priority</span>
            <span class='ml-2'>Severity: ${r.severity?.level || ''}</span>
          </div>
          <div class='text-sm text-gray-700 mt-2'>${r.description || 'No description'}</div>
          <button class='mt-3 text-sm text-blue-600 hover:text-blue-700' onclick='showMilestones("${r.id}")'>View Updates & Milestones →</button>
        </div>
      </div>
      <div id='milestones-${r.id}' class='mt-3 hidden border-t pt-3'></div>
    `;
    list.appendChild(card);
  });

  if (bounds.length) map.fitBounds(bounds, { padding: [30, 30] });
}

async function showMilestones(reportId) {
  const container = document.getElementById(`milestones-${reportId}`);
  if (container.classList.contains('hidden')) {
    container.classList.remove('hidden');
    container.innerHTML = '<div class="text-sm text-gray-500">Loading milestones...</div>';
    const milestones = await loadMilestones(reportId);
    
    if (milestones.length === 0) {
      container.innerHTML = '<div class="text-sm text-gray-500">No updates yet. Check back soon!</div>';
      return;
    }

    container.innerHTML = `
      <div class="space-y-2">
        <div class="font-medium text-sm mb-2">Updates & Milestones:</div>
        ${milestones.map(m => `
          <div class="border-l-2 border-blue-500 pl-3 py-2">
            <div class="font-medium text-sm">${m.title}</div>
            <div class="text-xs text-gray-600 mt-1">${m.description}</div>
            <div class="text-xs text-gray-500 mt-1">${new Date(m.createdAt).toLocaleString()}</div>
          </div>
        `).join('')}
      </div>
    `;
  } else {
    container.classList.add('hidden');
  }
}

async function loadReports() {
  try {
    const res = await fetch('/api/user/reports');
    const json = await res.json();
    if (json.success) {
      renderReports(json.reports || []);
    }
  } catch (e) {
    console.error('Failed to load reports', e);
  }
}

window.showMilestones = showMilestones;

window.addEventListener('DOMContentLoaded', () => {
  initMap();
  loadReports();
  // Auto-refresh every 30 seconds
  setInterval(loadReports, 30000);
});

