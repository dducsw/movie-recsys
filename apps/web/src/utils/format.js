// Shared formatting helpers. Single source of truth for movie display.

export const GENRES = [
  'Action', 'Adventure', 'Animation', 'Comedy', 'Crime', 'Drama',
  'Fantasy', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller',
];

const PLACEHOLDER_POSTER = 'https://via.placeholder.com/300x450?text=No+Poster';

export const formatRating = (voteAverage) => {
  const stars = Number(voteAverage || 0) / 2;
  return stars.toFixed(1);
};

export const posterUrl = (url) => {
  if (!url || url.includes('placeholder')) return PLACEHOLDER_POSTER;
  return url;
};

export const genreList = (genres) => {
  if (!genres) return [];
  return String(genres).split('|').filter(Boolean);
};

export const genreLabel = (genres) => genreList(genres).join(' • ');

const SAFE_ID = /^[a-zA-Z0-9_-]{6,20}$/;

export const getEmbedUrl = (videoId) => {
  if (!videoId) return '';
  const id = String(videoId).trim();
  if (id.includes('youtube.com/embed/') || id.includes('youtube-nocookie.com/embed/')) return id;
  const watch = id.match(/[?&]v=([a-zA-Z0-9_-]{6,20})/);
  const short = id.match(/youtu\.be\/([a-zA-Z0-9_-]{6,20})/);
  const raw = watch?.[1] || short?.[1] || id;
  if (!SAFE_ID.test(raw)) return '';
  return `https://www.youtube-nocookie.com/embed/${raw}`;
};

export const timeAgo = (iso) => {
  if (!iso) return '';
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
};

const AVATAR_COLORS = ['#e50914', '#221f1f', '#b20710', '#564d4d', '#0080ff'];

export const avatarColor = (name) => {
  let hash = 0;
  for (const c of String(name || 'U')) hash = (hash * 31 + c.charCodeAt(0)) >>> 0;
  return AVATAR_COLORS[hash % AVATAR_COLORS.length];
};
