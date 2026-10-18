// Minimal i18n: vi default, en fallback. No dependency.

const STRINGS = {
  vi: {
    showing: (n) => `Hiển thị ${n} phim`,
    noMovies: 'Không tìm thấy phim',
    noMoviesDesc: 'Thử đổi từ khóa, thể loại, hoặc xem trending.',
    prev: 'Trang trước',
    next: 'Trang sau',
    showMore: 'Xem thêm',
    loading: 'Đang tải...',
  },
  en: {
    showing: (n) => `Showing ${n} ${n === 1 ? 'movie' : 'movies'}`,
    noMovies: 'No movies found',
    noMoviesDesc: 'Try adjusting your search query, genre tags, or explore our trending catalog.',
    prev: 'Previous',
    next: 'Next',
    showMore: 'Show more',
    loading: 'Loading...',
  },
};

export const getLang = () =>
  (typeof localStorage !== 'undefined' && localStorage.getItem('movienex_lang') === 'en' ? 'en' : 'vi');

export const setLang = (lang) => {
  try {
    localStorage.setItem('movienex_lang', lang === 'en' ? 'en' : 'vi');
  } catch { /* ignore */ }
};

export const t = (key, ...args) => {
  const lang = getLang();
  const entry = STRINGS[lang][key] ?? STRINGS.vi[key];
  return typeof entry === 'function' ? entry(...args) : entry;
};
