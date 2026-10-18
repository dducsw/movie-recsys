import http from 'k6/http';
import { check, sleep } from 'k6';

// k6 SLA check: p95 <= 50ms is aspirational for warm cache; gate here is p95 < 300ms
// Usage: k6 run benchmarks/k6_recsys.js
export const options = {
  vus: 20,
  duration: '60s',
  thresholds: {
    http_req_duration: ['p(95)<300'],
    http_req_failed: ['rate<0.01'],
  },
};

const BASE = __ENV.API_BASE || 'http://localhost:8000';

export default function () {
  const r1 = http.get(`${BASE}/api/movies/trending?limit=20`);
  check(r1, { 'trending 200': (r) => r.status === 200 });

  const r2 = http.get(`${BASE}/api/recsys/recommendations?limit=20`);
  check(r2, { 'recs 200/401': (r) => [200, 401, 422].includes(r.status) });
  sleep(1);
}
