export const API_BASE_URL = import.meta.env?.VITE_API_URL || 'http://localhost:8000/api';

// In-memory token storage (XSS-safe: access token is never persisted in localStorage/sessionStorage)
let inMemoryAccessToken = null;

export const setAccessToken = (token) => {
  inMemoryAccessToken = token;
};

export const getAccessToken = () => {
  return inMemoryAccessToken;
};

export const resetSessionId = () => {
  const newSid = crypto.randomUUID();
  sessionStorage.setItem('mn_sid', newSid);
  return newSid;
};

export const getOrCreateSessionId = () => {
  const KEY = 'mn_sid';
  let sid = sessionStorage.getItem(KEY);
  if (!sid) {
    sid = crypto.randomUUID();
    sessionStorage.setItem(KEY, sid);
  }
  return sid;
};

let refreshPromise = null;

export const refreshAccessToken = async () => {
  if (refreshPromise) return refreshPromise;

  refreshPromise = (async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/auth/refresh`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'X-Requested-With': 'XMLHttpRequest',
        },
        credentials: 'include',
      });

      if (!response.ok) {
        setAccessToken(null);
        return null;
      }

      const data = await response.json();
      if (data && data.access_token) {
        setAccessToken(data.access_token);
        return data;
      }
      return null;
    } catch {
      setAccessToken(null);
      return null;
    } finally {
      refreshPromise = null;
    }
  })();

  return refreshPromise;
};

export const apiFetch = async (path, options = {}) => {
  const token = getAccessToken();
  const headers = {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest', // Anti-CSRF protection
    'X-Session-Id': getOrCreateSessionId(),
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...(options.headers || {})
  };

  let response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: 'include'
  });

  // Automatic token refresh interceptor on 401
  if (response.status === 401 && !path.startsWith('/auth/refresh') && !path.startsWith('/auth/login')) {
    const refreshed = await refreshAccessToken();
    if (refreshed && refreshed.access_token) {
      headers['Authorization'] = `Bearer ${refreshed.access_token}`;
      response = await fetch(`${API_BASE_URL}${path}`, {
        ...options,
        headers,
        credentials: 'include'
      });
    }
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP Error ${response.status}`);
  }

  return response.json();
};

const trackingHeaders = () => {
  const token = getAccessToken();
  return {
    'Content-Type': 'application/json',
    'X-Requested-With': 'XMLHttpRequest',
    'X-Session-Id': getOrCreateSessionId(),
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
  };
};

export const trackClick = (movieId, source, position = null) => {
  fetch(`${API_BASE_URL}/events/click`, {
    method: 'POST',
    headers: trackingHeaders(),
    credentials: 'include',
    keepalive: true,
    body: JSON.stringify({ movie_id: movieId, source, position }),
  }).catch(() => {});
};

export const trackDwell = (movieId, dwellTimeMs, source = 'listing', position = null) => {
  if (!movieId || !dwellTimeMs || dwellTimeMs < 500) return;
  fetch(`${API_BASE_URL}/events/dwell`, {
    method: 'POST',
    headers: trackingHeaders(),
    credentials: 'include',
    keepalive: true,
    body: JSON.stringify({ movie_id: movieId, dwell_time_ms: Math.round(dwellTimeMs), source, position }),
  }).catch(() => {});
};

export const trackImpressions = (items) => {
  if (!items || items.length === 0) return;
  fetch(`${API_BASE_URL}/events/impression`, {
    method: 'POST',
    headers: trackingHeaders(),
    credentials: 'include',
    keepalive: true,
    body: JSON.stringify({
      impressions: items.map(item => ({
        movie_id: item.movieId,
        source: item.source || 'listing',
        position: item.position ?? null
      }))
    }),
  }).catch(() => {});
};

let impressionBuffer = [];
let impressionTimeout = null;

const flushImpressions = () => {
  if (impressionTimeout) {
    clearTimeout(impressionTimeout);
    impressionTimeout = null;
  }
  if (impressionBuffer.length === 0) return;
  const toSend = [...impressionBuffer];
  impressionBuffer = [];
  // Deduplicate by movieId + source + position within the batch window
  const seen = new Set();
  const uniqueItems = [];
  for (const imp of toSend) {
    const key = `${imp.movieId}-${imp.source || 'listing'}-${imp.position ?? ''}`;
    if (!seen.has(key)) {
      seen.add(key);
      uniqueItems.push(imp);
    }
  }
  trackImpressions(uniqueItems.slice(0, 50));
};

export const queueImpression = (item) => {
  if (!item || !item.movieId) return;
  impressionBuffer.push(item);
  if (!impressionTimeout) {
    impressionTimeout = setTimeout(flushImpressions, 400);
  }
};

// Flush pending impressions on navigate/close (was lost before).
if (typeof window !== 'undefined') {
  window.addEventListener('pagehide', flushImpressions);
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'hidden') flushImpressions();
  });
}
