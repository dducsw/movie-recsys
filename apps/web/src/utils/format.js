// Shared display helpers for movies.

// Inline SVG placeholder — no external service dependency.
const PLACEHOLDER_POSTER =
  "data:image/svg+xml," +
  encodeURIComponent(
    '<svg xmlns="http://www.w3.org/2000/svg" width="300" height="450"><rect width="100%" height="100%" fill="#1a2234"/><text x="150" y="228" fill="#64748b" font-family="sans-serif" font-size="16" text-anchor="middle">No Poster</text></svg>'
  );

export const posterUrl = (url) => {
  if (!url || url.includes('placeholder')) return PLACEHOLDER_POSTER;
  return url;
};

// vote_average (0–10 TMDB) -> 0–5 scale for display.
export const formatRating = (voteAverage) => {
  const stars = Number(voteAverage || 0) / 2;
  return stars ? stars.toFixed(1) : null; // null = không hiện thay vì bịa số
};

export const releaseYear = (releaseDate) => {
  const date = releaseDate ? String(releaseDate) : '';
  return date.match(/\d{4}/)?.[0] || null;
};

const SAFE_ID = /^[a-zA-Z0-9_-]{6,20}$/;

// Accepts a trailer URL / embed URL / raw video id, returns a youtube-nocookie
// embed URL, or '' if not safe. Caller decides the fallback.
export const getEmbedUrl = (trailerUrl) => {
  if (!trailerUrl) return '';
  const raw = String(trailerUrl).trim();
  if (raw.includes('/embed/')) {
    const id = raw.split('/embed/')[1]?.split('?')[0];
    return id && SAFE_ID.test(id) ? `https://www.youtube-nocookie.com/embed/${id}` : '';
  }
  const watch = raw.match(/[?&]v=([a-zA-Z0-9_-]{6,20})/);
  const short = raw.match(/youtu\.be\/([a-zA-Z0-9_-]{6,20})/);
  const id = watch?.[1] || short?.[1];
  return id && SAFE_ID.test(id) ? `https://www.youtube-nocookie.com/embed/${id}` : '';
};
