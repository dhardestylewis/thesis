// Smoke test for live_friction_map.html.
//
//   node WebMap_Prototype/smoke_test_map.mjs            both backends
//   node WebMap_Prototype/smoke_test_map.mjs api        estimator on :8001 only
//   node WebMap_Prototype/smoke_test_map.mjs static     range-served files only
//
// Runs the page's own script with a stub DOM and prints the panel values it would
// display. Catches the class of bug where the data source answers but the page
// throws while rendering, which surfaces only as "unavailable" in the UI.
//
// The static case starts a local range-capable server over the packed grid, which
// is the same shape the public object storage serves, and checks that both
// backends produce identical panel values.
import fs from 'node:fs';
import http from 'node:http';
import path from 'node:path';

const DATA_DIR = 'C:/Users/dhl/data/Thesis/thesis/Data/Zoning_Cases';
const STATIC_FILES = {
  '/austin_friction_grid.f32': path.join(DATA_DIR, 'austin_friction_grid.f32'),
  '/austin_base_geometries.fgb': path.join(DATA_DIR, 'austin_base_geometries_cached.fgb'),
};

const html = fs.readFileSync(new URL('./live_friction_map.html', import.meta.url), 'utf8');
const script = html.split('<script>').pop().split('</script>')[0];

function startRangeServer() {
  const server = http.createServer((req, res) => {
    const file = STATIC_FILES[req.url.split('?')[0]];
    if (!file || !fs.existsSync(file)) return res.writeHead(404).end();
    const size = fs.statSync(file).size;
    const m = /^bytes=(\d+)-(\d+)$/.exec(req.headers.range || '');
    if (!m) {
      res.writeHead(200, { 'Content-Length': size, 'Accept-Ranges': 'bytes' });
      return fs.createReadStream(file).pipe(res);
    }
    const [start, end] = [Number(m[1]), Math.min(Number(m[2]), size - 1)];
    res.writeHead(206, {
      'Content-Range': `bytes ${start}-${end}/${size}`,
      'Content-Length': end - start + 1,
      'Accept-Ranges': 'bytes',
    });
    fs.createReadStream(file, { start, end }).pipe(res);
  });
  return new Promise(resolve => server.listen(0, '127.0.0.1', () => resolve(server)));
}

async function run(search) {
  const els = {};
  const mkEl = id => (els[id] ??= {
    id, style: {}, textContent: '', innerHTML: '', checked: false, value: '',
    classList: { toggle() {}, add() {}, remove() {} },
    addEventListener() {}, dataset: {},
  });

  let loadHandler = null;
  globalThis.document = { getElementById: mkEl, querySelectorAll: () => [] };
  globalThis.location = { search, protocol: 'http:', hostname: '127.0.0.1' };
  // keep the real performance object: undici needs markResourceTiming
  globalThis.maplibregl = { Map: class { on(ev, cb) { if (ev === 'load') loadHandler = cb; } addControl() {} } };
  globalThis.deck = { MapboxOverlay: class { setProps() {} }, GeoJsonLayer: class {} };
  globalThis.flatgeobuf = {
    deserialize: async function* () {
      for (let i = 0; i < 3; i++) yield { properties: { prop_id: i, basezone: 'SF-3', basezone_h: 30 } };
    },
  };

  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, opts) => {
    // Geometry parsing is stubbed above; only its transport differs per backend
    if (String(url).includes('geometries') || String(url).endsWith('.fgb')) return { ok: true, body: null };
    return realFetch(url, opts);
  };

  eval(script);
  await loadHandler();
  await new Promise(r => setTimeout(r, 300));

  return {
    status: els['api-status']?.textContent,
    values: ['stat-parcels', 'stat-delay', 'stat-delay95', 'stat-attr', 'stat-killed']
      .map(id => `${id}=${els[id]?.textContent || '(never set)'}`),
  };
}

const which = process.argv[2] || 'both';
const results = {};

if (which === 'api' || which === 'both') {
  results.api = await run('');
  console.log('api backend   :', results.api.status);
  results.api.values.forEach(v => console.log('   ', v));
}

if (which === 'static' || which === 'both') {
  const server = await startRangeServer();
  const { port } = server.address();
  results.static = await run(`?backend=static&data=http://127.0.0.1:${port}`);
  server.close();
  console.log('static backend:', results.static.status);
  results.static.values.forEach(v => console.log('   ', v));
}

if (results.api && results.static) {
  const same = JSON.stringify(results.api.values) === JSON.stringify(results.static.values);
  console.log(same
    ? '\nBackends agree: the published page will show what the local estimator shows.'
    : '\nMISMATCH between backends. Investigate before deploying.');
  if (!same) process.exit(1);
}
