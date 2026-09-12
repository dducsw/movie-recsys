import React, { useState, useEffect } from 'react';
import { 
  MessageSquare, 
  ThumbsUp, 
  Reply, 
  Trash2, 
  Send, 
  User, 
  LogIn, 
  CornerDownRight, 
  ChevronDown, 
  ChevronUp,
  Clock
} from 'lucide-react';
import { API_BASE_URL } from '../../api/client';
import { useAuth } from '../../context/AuthContext';
import './CommentSection.css';

function formatRelativeTime(isoString) {
  if (!isoString) return 'Vừa xong';
  try {
    let dateStr = String(isoString).trim();
    // Nếu chuỗi datetime không có múi giờ (không có Z và không có +/- offset), ép về UTC
    if (!dateStr.endsWith('Z') && !/[+-]\d{2}(:\d{2})?$/.test(dateStr)) {
      dateStr = dateStr.replace(' ', 'T') + 'Z';
    }
    const now = new Date();
    const past = new Date(dateStr);
    const diffSeconds = Math.floor((now.getTime() - past.getTime()) / 1000);

    if (diffSeconds < 60) return 'Vừa xong';
    const diffMinutes = Math.floor(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes} phút trước`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours} giờ trước`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 30) return `${diffDays} ngày trước`;
    return past.toLocaleDateString('vi-VN');
  } catch (e) {
    return 'Vừa xong';
  }
}

// Generate consistent avatar color based on username
function getAvatarColor(name = '') {
  const colors = [
    '#3b82f6', '#10b981', '#f59e0b', '#ef4444', 
    '#8b5cf6', '#ec4899', '#06b6d4', '#14b8a6'
  ];
  let hash = 0;
  for (let i = 0; i < name.length; i++) {
    hash = name.charCodeAt(i) + ((hash << 5) - hash);
  }
  return colors[Math.abs(hash) % colors.length];
}

export default function CommentSection({ movieId, movieTitle }) {
  const { user, openAuthModal } = useAuth();
  const [comments, setComments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [newCommentText, setNewCommentText] = useState('');
  const [replyingToId, setReplyingToId] = useState(null);
  const [replyText, setReplyText] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [showRepliesMap, setShowRepliesMap] = useState({});

  // Fetch comments for this movie
  const fetchComments = async () => {
    try {
      const token = localStorage.getItem('auth_token');
      const headers = token ? { 'Authorization': `Bearer ${token}` } : {};
      const res = await fetch(`${API_BASE_URL}/movies/${movieId}/comments`, { headers });
      if (res.ok) {
        const data = await res.json();
        setComments(data.comments || []);
      }
    } catch (err) {
      console.error('Failed to fetch comments:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (movieId) {
      setLoading(true);
      fetchComments();
    }
  }, [movieId]);

  // Handle post new top-level comment
  const handlePostComment = async (e) => {
    e.preventDefault();
    if (!newCommentText.trim() || submitting) return;

    if (!user) {
      openAuthModal();
      return;
    }

    setSubmitting(true);
    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(`${API_BASE_URL}/movies/${movieId}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ content: newCommentText.trim() })
      });

      if (res.ok) {
        const data = await res.json();
        setComments([data.comment, ...comments]);
        setNewCommentText('');
      } else {
        const errData = await res.json();
        alert(errData.detail || 'Không thể đăng bình luận.');
      }
    } catch (err) {
      alert('Lỗi kết nối khi đăng bình luận.');
    } finally {
      setSubmitting(false);
    }
  };

  // Handle post reply to a comment
  const handlePostReply = async (parentId) => {
    if (!replyText.trim() || submitting) return;

    if (!user) {
      openAuthModal();
      return;
    }

    setSubmitting(true);
    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(`${API_BASE_URL}/movies/${movieId}/comments`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ content: replyText.trim(), parent_id: parentId })
      });

      if (res.ok) {
        const data = await res.json();
        // Insert new reply under parent
        setComments(comments.map(c => {
          if (c.id === parentId) {
            return {
              ...c,
              replies: [...(c.replies || []), data.comment]
            };
          }
          return c;
        }));
        setReplyText('');
        setReplyingToId(null);
        // Ensure replies are expanded
        setShowRepliesMap(prev => ({ ...prev, [parentId]: true }));
      } else {
        const errData = await res.json();
        alert(errData.detail || 'Không thể gửi câu trả lời.');
      }
    } catch (err) {
      alert('Lỗi kết nối khi gửi phản hồi.');
    } finally {
      setSubmitting(false);
    }
  };

  // Handle Like/Unlike
  const handleToggleLike = async (commentId, isReply = false, parentId = null) => {
    if (!user) {
      openAuthModal();
      return;
    }

    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(`${API_BASE_URL}/comments/${commentId}/like`, {
        method: 'POST',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (res.ok) {
        const data = await res.json();
        setComments(comments.map(c => {
          if (!isReply && c.id === commentId) {
            return { ...c, is_liked: data.is_liked, likes_count: data.likes_count };
          }
          if (isReply && c.id === parentId) {
            return {
              ...c,
              replies: c.replies.map(r => 
                r.id === commentId ? { ...r, is_liked: data.is_liked, likes_count: data.likes_count } : r
              )
            };
          }
          return c;
        }));
      }
    } catch (err) {
      console.error('Like toggle error:', err);
    }
  };

  // Handle Delete Comment
  const handleDeleteComment = async (commentId, isReply = false, parentId = null) => {
    if (!window.confirm('Bạn có chắc chắn muốn xóa bình luận này không?')) return;

    try {
      const token = localStorage.getItem('auth_token');
      const res = await fetch(`${API_BASE_URL}/comments/${commentId}`, {
        method: 'DELETE',
        headers: { 'Authorization': `Bearer ${token}` }
      });

      if (res.ok) {
        if (!isReply) {
          setComments(comments.filter(c => c.id !== commentId));
        } else {
          setComments(comments.map(c => {
            if (c.id === parentId) {
              return { ...c, replies: c.replies.filter(r => r.id !== commentId) };
            }
            return c;
          }));
        }
      } else {
        const errData = await res.json();
        alert(errData.detail || 'Không thể xóa bình luận.');
      }
    } catch (err) {
      alert('Lỗi kết nối khi xóa bình luận.');
    }
  };

  const toggleReplies = (commentId) => {
    setShowRepliesMap(prev => ({
      ...prev,
      [commentId]: !prev[commentId]
    }));
  };

  const totalCommentsCount = comments.reduce((sum, c) => sum + 1 + (c.replies?.length || 0), 0);

  return (
    <section className="fb-comments-section" id="movie-comments-section">
      <div className="fb-comments-header">
        <div className="fb-comments-title-row">
          <MessageSquare className="w-5 h-5 text-blue-500" />
          <h3 className="fb-comments-title">Bình Luận Khán Giả</h3>
          <span className="fb-comments-badge">{totalCommentsCount}</span>
        </div>
        <p className="fb-comments-subtitle">
          Thảo luận, đánh giá và chia sẻ cảm nghĩ của bạn về bộ phim <strong>{movieTitle}</strong>
        </p>
      </div>

      {/* Write Comment Box */}
      <div className="fb-create-comment-card">
        {user ? (
          <div className="fb-comment-input-layout">
            <div 
              className="fb-user-avatar" 
              style={{ backgroundColor: getAvatarColor(user.username || user.email) }}
            >
              {(user.username || user.email || 'U')[0].toUpperCase()}
            </div>
            <form onSubmit={handlePostComment} className="fb-comment-form">
              <textarea
                className="fb-comment-textarea"
                placeholder="Viết bình luận công khai của bạn về bộ phim..."
                value={newCommentText}
                onChange={(e) => setNewCommentText(e.target.value)}
                rows={2}
                disabled={submitting}
              />
              <div className="fb-comment-form-footer">
                <span className="fb-input-hint">Nhấn Đăng để chia sẻ cảm nghĩ của bạn</span>
                <button
                  type="submit"
                  className="fb-btn-submit"
                  disabled={submitting || !newCommentText.trim()}
                >
                  <Send className="w-4 h-4" />
                  <span>{submitting ? 'Đang gửi...' : 'Đăng'}</span>
                </button>
              </div>
            </form>
          </div>
        ) : (
          <div className="fb-auth-prompt">
            <div className="fb-auth-prompt-info">
              <div className="fb-prompt-avatar-placeholder">
                <User className="w-5 h-5" />
              </div>
              <span>Đăng nhập để tham gia bình luận và chia sẻ cảm xúc cùng cộng đồng!</span>
            </div>
            <button className="fb-btn-login-prompt" onClick={openAuthModal}>
              <LogIn className="w-4 h-4" />
              <span>Đăng Nhập Ngay</span>
            </button>
          </div>
        )}
      </div>

      {/* Comments List */}
      <div className="fb-comments-list">
        {loading ? (
          <div className="fb-comments-loading">
            <div className="spinner" />
            <span>Đang tải bình luận...</span>
          </div>
        ) : comments.length === 0 ? (
          <div className="fb-comments-empty">
            <MessageSquare className="w-10 h-10 text-muted opacity-40 mb-2" />
            <p className="font-medium text-foreground">Chưa có bình luận nào</p>
            <p className="text-sm text-muted">Hãy là người đầu tiên để lại cảm nghĩ về bộ phim này!</p>
          </div>
        ) : (
          comments.map((comment) => {
            const hasReplies = comment.replies && comment.replies.length > 0;
            const isReplying = replyingToId === comment.id;
            const areRepliesVisible = showRepliesMap[comment.id] !== false; // Default expanded

            return (
              <div key={comment.id} className="fb-comment-item">
                <div className="fb-comment-main-row">
                  {/* Author Avatar */}
                  <div 
                    className="fb-user-avatar"
                    style={{ backgroundColor: getAvatarColor(comment.username) }}
                  >
                    {(comment.username || 'U')[0].toUpperCase()}
                  </div>

                  {/* Comment Bubble Content */}
                  <div className="fb-comment-body">
                    <div className="fb-comment-bubble">
                      <div className="fb-author-row">
                        <span className="fb-author-name">{comment.username}</span>
                        {comment.user_id === user?.id && (
                          <span className="fb-badge-author">Bạn</span>
                        )}
                      </div>
                      <p className="fb-comment-text">{comment.content}</p>

                      {/* Floating like counter on bubble */}
                      {comment.likes_count > 0 && (
                        <div className="fb-bubble-like-counter">
                          <ThumbsUp className="w-3 h-3 fill-current text-blue-500" />
                          <span>{comment.likes_count}</span>
                        </div>
                      )}
                    </div>

                    {/* Action Row */}
                    <div className="fb-comment-actions">
                      <button 
                        className={`fb-action-btn ${comment.is_liked ? 'liked' : ''}`}
                        onClick={() => handleToggleLike(comment.id, false)}
                      >
                        <ThumbsUp className="w-3.5 h-3.5" fill={comment.is_liked ? "currentColor" : "none"} />
                        <span>Thích</span>
                      </button>

                      <button 
                        className="fb-action-btn"
                        onClick={() => {
                          setReplyingToId(isReplying ? null : comment.id);
                          setReplyText('');
                        }}
                      >
                        <Reply className="w-3.5 h-3.5" />
                        <span>Phản hồi</span>
                      </button>

                      {user && user.id === comment.user_id && (
                        <button 
                          className="fb-action-btn delete"
                          onClick={() => handleDeleteComment(comment.id, false)}
                          title="Xóa bình luận này"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                          <span>Xóa</span>
                        </button>
                      )}

                      <span className="fb-comment-time">
                        <Clock className="w-3 h-3 inline mr-1 opacity-70" />
                        {formatRelativeTime(comment.created_at)}
                      </span>
                    </div>

                    {/* Inline Reply Input Form */}
                    {isReplying && (
                      <div className="fb-reply-box animate-in fade-in duration-150">
                        <CornerDownRight className="w-4 h-4 text-blue-400 mt-2" />
                        <div className="fb-reply-input-wrapper">
                          <input
                            type="text"
                            placeholder={`Trả lời ${comment.username}...`}
                            value={replyText}
                            onChange={(e) => setReplyText(e.target.value)}
                            className="fb-reply-input"
                            autoFocus
                            onKeyDown={(e) => {
                              if (e.key === 'Enter' && !e.shiftKey) {
                                e.preventDefault();
                                handlePostReply(comment.id);
                              }
                            }}
                          />
                          <button
                            className="fb-btn-send-reply"
                            disabled={!replyText.trim() || submitting}
                            onClick={() => handlePostReply(comment.id)}
                          >
                            <Send className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </div>
                    )}

                    {/* Threaded Replies */}
                    {hasReplies && (
                      <div className="fb-replies-container">
                        <button 
                          className="fb-toggle-replies-btn"
                          onClick={() => toggleReplies(comment.id)}
                        >
                          {areRepliesVisible ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                          <span>
                            {areRepliesVisible 
                              ? `Ẩn ${comment.replies.length} phản hồi` 
                              : `Xem ${comment.replies.length} phản hồi`}
                          </span>
                        </button>

                        {areRepliesVisible && (
                          <div className="fb-replies-list">
                            {comment.replies.map((reply) => (
                              <div key={reply.id} className="fb-reply-item">
                                <div 
                                  className="fb-user-avatar small"
                                  style={{ backgroundColor: getAvatarColor(reply.username) }}
                                >
                                  {(reply.username || 'U')[0].toUpperCase()}
                                </div>
                                <div className="fb-comment-body">
                                  <div className="fb-comment-bubble reply">
                                    <div className="fb-author-row">
                                      <span className="fb-author-name">{reply.username}</span>
                                      {reply.user_id === user?.id && (
                                        <span className="fb-badge-author">Bạn</span>
                                      )}
                                    </div>
                                    <p className="fb-comment-text">{reply.content}</p>

                                    {reply.likes_count > 0 && (
                                      <div className="fb-bubble-like-counter">
                                        <ThumbsUp className="w-2.5 h-2.5 fill-current text-blue-500" />
                                        <span>{reply.likes_count}</span>
                                      </div>
                                    )}
                                  </div>

                                  <div className="fb-comment-actions">
                                    <button 
                                      className={`fb-action-btn ${reply.is_liked ? 'liked' : ''}`}
                                      onClick={() => handleToggleLike(reply.id, true, comment.id)}
                                    >
                                      <ThumbsUp className="w-3 h-3" fill={reply.is_liked ? "currentColor" : "none"} />
                                      <span>Thích</span>
                                    </button>

                                    {user && user.id === reply.user_id && (
                                      <button 
                                        className="fb-action-btn delete"
                                        onClick={() => handleDeleteComment(reply.id, true, comment.id)}
                                      >
                                        <Trash2 className="w-3 h-3" />
                                        <span>Xóa</span>
                                      </button>
                                    )}

                                    <span className="fb-comment-time">
                                      {formatRelativeTime(reply.created_at)}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </section>
  );
}
