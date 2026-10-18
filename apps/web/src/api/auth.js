// Auth API helpers. Single place for login/register headers.

import { API_BASE_URL, setAccessToken, getOrCreateSessionId } from './client';

const baseHeaders = () => ({
  'Content-Type': 'application/json',
  'X-Requested-With': 'XMLHttpRequest',
  'X-Session-Id': getOrCreateSessionId(),
});

async function handleAuthResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `HTTP Error ${res.status}`);
  }
  if (data?.access_token) setAccessToken(data.access_token);
  return data;
}

export const loginApi = (payload) =>
  fetch(`${API_BASE_URL}/auth/login`, {
    method: 'POST',
    headers: baseHeaders(),
    credentials: 'include',
    body: JSON.stringify(payload),
  }).then(handleAuthResponse);

export const registerApi = (payload) =>
  fetch(`${API_BASE_URL}/auth/register`, {
    method: 'POST',
    headers: baseHeaders(),
    credentials: 'include',
    body: JSON.stringify(payload),
  }).then(handleAuthResponse);
