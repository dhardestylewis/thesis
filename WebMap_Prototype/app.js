const map = new maplibregl.Map({
    container: 'map',
    style: 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json',
    center: [-97.7431, 30.2672], // Austin, TX
    zoom: 11,
    pitch: 60,
    bearing: -20,
    antialias: true
});

// UI Elements
const heightSlider = document.getElementById('height-slider');
const heightValue = document.getElementById('height-value');
const delayMetric = document.querySelector('#metric-delay .metric-value');
const withdrawalMetric = document.querySelector('#metric-withdrawal .metric-value');
const attritionMetric = document.querySelector('#metric-attrition .metric-value');

// Causal Engine (Deterministic Thresholds from thesis)
function calculateCausalMetrics(income, lat, lng, requestedHeight) {
    let delay = 0;
    let withdrawalProb = 0.0;
    let attrition = 0;

    // The Skyscraper Override
    if (requestedHeight > 20) {
        delay = 836;
        withdrawalProb = 0.10;
        attrition = 0; // Negative CATE theoretically, but 0 for display
    } 
    // The Missing Middle Trap
    else {
        // The Far-West Fast-Track
        if (lng < -97.82) {
            delay = 98; // Very fast or immediate withdrawal
            withdrawalProb = 0.85; 
            attrition = 15;
        }
        // The Gentrification War Zone (East/North Austin)
        else if (income < 62120) {
            delay = Math.floor(Math.random() * 250) + 400; // 400 - 650 days (Realistic bureaucratic friction for survivors)
            withdrawalProb = 0.95;
            attrition = requestedHeight * 0.8; // Shaves most of it
        }
        // General Missing Middle
        else {
            delay = 300;
            withdrawalProb = 0.60;
            attrition = 12;
        }
    }

    return {
        delay: Math.round(delay),
        withdrawal: Math.round(withdrawalProb * 100),
        attrition: Math.round(Math.min(attrition, requestedHeight))
    };
}

let hoveredStateId = null;
let currentGeoJSON = null;

map.on('load', () => {
    // Load real Austin Data from actual historical zoning cases
    fetch('real_austin_data.geojson')
        .then(response => response.json())
        .then(data => {
            currentGeoJSON = data;
            
            // Add Source
            map.addSource('austin-parcels', {
                'type': 'geojson',
                'data': data,
                'generateId': true
            });

            // Add 3D Extrusion Layer
            map.addLayer({
                'id': 'parcels-3d',
                'type': 'fill-extrusion',
                'source': 'austin-parcels',
                'paint': {
                    // Color maps to Withdrawal Probability (Cyan = 0%, Red = 100%)
                    'fill-extrusion-color': [
                        'interpolate',
                        ['linear'],
                        ['get', 'Withdrawal_Prob'],
                        0, '#00f0ff',
                        50, '#ffaa00',
                        100, '#ff003c'
                    ],
                    // Height maps to Delay
                    'fill-extrusion-height': ['get', 'Delay'],
                    // Base is 0
                    'fill-extrusion-base': 0,
                    'fill-extrusion-opacity': 0.85
                }
            });

            // Add highlight layer for hover
            map.addLayer({
                'id': 'parcels-highlight',
                'type': 'line',
                'source': 'austin-parcels',
                'paint': {
                    'line-color': '#ffffff',
                    'line-width': [
                        'case',
                        ['boolean', ['feature-state', 'hover'], false],
                        3,
                        0
                    ]
                }
            });

            updateMapData();
        });

    // Hover Effect & HUD Update
    map.on('mousemove', 'parcels-3d', (e) => {
        if (e.features.length > 0) {
            if (hoveredStateId !== null) {
                map.setFeatureState(
                    { source: 'austin-parcels', id: hoveredStateId },
                    { hover: false }
                );
            }
            hoveredStateId = e.features[0].id;
            map.setFeatureState(
                { source: 'austin-parcels', id: hoveredStateId },
                { hover: true }
            );

            // Update UI Sidebar with hovered tract stats
            const props = e.features[0].properties;
            delayMetric.innerText = `${props.Delay} Days`;
            withdrawalMetric.innerText = `${props.Withdrawal_Prob}%`;
            attritionMetric.innerText = `${props.Attrition} ft`;
            
            // Update Colors based on severity
            delayMetric.style.color = props.Delay > 1000 ? 'var(--accent-red)' : (props.Delay > 500 ? '#ffaa00' : 'var(--text-main)');
            withdrawalMetric.style.color = props.Withdrawal_Prob > 75 ? 'var(--accent-red)' : 'var(--text-main)';
        }
    });

    map.on('mouseleave', 'parcels-3d', () => {
        if (hoveredStateId !== null) {
            map.setFeatureState(
                { source: 'austin-parcels', id: hoveredStateId },
                { hover: false }
            );
        }
        hoveredStateId = null;
        
        // Reset UI Sidebar
        delayMetric.innerText = `-- Days`;
        withdrawalMetric.innerText = `-- %`;
        attritionMetric.innerText = `-- ft`;
        delayMetric.style.color = 'var(--text-main)';
        withdrawalMetric.style.color = 'var(--text-main)';
    });
});

// Slider Event
heightSlider.addEventListener('input', (e) => {
    const val = parseInt(e.target.value);
    heightValue.innerText = `${val} ft`;
    updateMapData();
});

// Recompute all metrics when slider changes
function updateMapData() {
    if (!currentGeoJSON) return;
    
    const requestedHeight = parseInt(heightSlider.value);
    
    currentGeoJSON.features.forEach(feat => {
        const props = feat.properties;
        const metrics = calculateCausalMetrics(
            props.Median_Income, 
            props.Latitude, 
            props.Longitude, 
            requestedHeight
        );
        
        props.Delay = metrics.delay;
        props.Withdrawal_Prob = metrics.withdrawal;
        props.Attrition = metrics.attrition;
    });

    map.getSource('austin-parcels').setData(currentGeoJSON);
}
