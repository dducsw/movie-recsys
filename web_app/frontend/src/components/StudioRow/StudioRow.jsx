import React from 'react';
import './StudioRow.css';

function StudioRow({ onSelectStudio }) {
  const curatedCategories = [
    {
      id: 'action_scifi',
      title: 'Action & Sci-Fi',
      query: 'Action',
      desc: 'Blockbusters & Adventures',
      icon: '🚀',
      gradient: 'linear-gradient(135deg, rgba(239, 68, 68, 0.15) 0%, rgba(168, 85, 247, 0.2) 100%)',
      border: 'rgba(239, 68, 68, 0.25)'
    },
    {
      id: 'comedy_fun',
      title: 'Comedy & Family',
      query: 'Comedy',
      desc: 'Laughs & Entertainment',
      icon: '🎭',
      gradient: 'linear-gradient(135deg, rgba(245, 158, 11, 0.15) 0%, rgba(234, 179, 8, 0.2) 100%)',
      border: 'rgba(245, 158, 11, 0.25)'
    },
    {
      id: 'drama_romance',
      title: 'Drama & Romance',
      query: 'Drama',
      desc: 'Emotional & Moving Stories',
      icon: '💖',
      gradient: 'linear-gradient(135deg, rgba(236, 72, 153, 0.15) 0%, rgba(168, 85, 247, 0.2) 100%)',
      border: 'rgba(236, 72, 153, 0.25)'
    },
    {
      id: 'horror_mystery',
      title: 'Horror & Thriller',
      query: 'Horror',
      desc: 'Suspense & Dark Mysteries',
      icon: '👻',
      gradient: 'linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(59, 130, 246, 0.2) 100%)',
      border: 'rgba(99, 102, 241, 0.25)'
    }
  ];

  return (
    <div className="cinemax-category-highlight-row">
      {curatedCategories.map((cat) => (
        <div 
          key={cat.id} 
          className="category-highlight-card"
          style={{ background: cat.gradient, borderColor: cat.border }}
          onClick={() => onSelectStudio && onSelectStudio(cat.query)}
        >
          <span className="cat-icon">{cat.icon}</span>
          <div className="cat-text-wrap">
            <span className="cat-title">{cat.title}</span>
            <span className="cat-desc">{cat.desc}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

export default StudioRow;
