export const API_BASE_URL = 'http://localhost:8000/api';

export const getOrCreateSessionId = () => {
  const KEY = 'movienex_sid';
  let sid = localStorage.getItem(KEY);
  if (!sid) {
    sid = crypto.randomUUID();
    localStorage.setItem(KEY, sid);
  }
  return sid;
};

export const apiFetch = async (path, options = {}) => {
  const token = localStorage.getItem('auth_token');
  const headers = {
    'Content-Type': 'application/json',
    'X-Session-Id': getOrCreateSessionId(),
    ...(token ? { 'Authorization': `Bearer ${token}` } : {}),
    ...(options.headers || {})
  };

  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...options,
    headers,
    credentials: 'include'
  });

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.detail || `HTTP Error ${response.status}`);
  }

  return response.json();
};

export const trackClick = (movieId, source, position = null) => {
  fetch(`${API_BASE_URL}/events/click`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': getOrCreateSessionId(),
    },
    credentials: 'include',
    body: JSON.stringify({ movie_id: movieId, source, position }),
  }).catch(() => {});
};

export const trackImpressions = (movieIds, source) => {
  if (!movieIds || movieIds.length === 0) return;
  fetch(`${API_BASE_URL}/events/impression`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'X-Session-Id': getOrCreateSessionId(),
    },
    credentials: 'include',
    body: JSON.stringify({ movie_ids: movieIds, source }),
  }).catch(() => {});
};
