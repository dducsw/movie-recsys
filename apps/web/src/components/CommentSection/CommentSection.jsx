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
import { apiFetch } from '../../api/client';
import { useAuth } from '../../context/AuthContext';
import './CommentSection.css';

function formatRelativeTime(isoString) {
  if (!isoString) return 'Just now';
  try {
    let dateStr = String(isoString).trim();
    // Nếu chuỗi datetime không có múi giờ (không có Z và không có +/- offset), ép về UTC
    if (!dateStr.endsWith('Z') && !/[+-]\d{2}(:\d{2})?$/.test(dateStr)) {
      dateStr = dateStr.replace(' ', 'T') + 'Z';
    }
    const now = new Date();
    const past = new Date(dateStr);
    const diffSeconds = Math.floor((now.getTime() - past.getTime()) / 1000);

    if (diffSeconds < 60) return 'Just now';
    const diffMinutes = Math.floor(diffSeconds / 60);
    if (diffMinutes < 60) return `${diffMinutes}m ago`;
    const diffHours = Math.floor(diffMinutes / 60);
    if (diffHours < 24) return `${diffHours}h ago`;
    const diffDays = Math.floor(diffHours / 24);
    if (diffDays < 30) return `${diffDays}d ago`;
    return past.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  } catch {
    return 'Just now';
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

  // Load comments mỗi khi đổi phim.
  useEffect(() => {
    const fetchComments = async () => {
      try {
        const data = await apiFetch(`/movies/${movieId}/comments`);
        setComments(data.comments || []);
      } catch (err) {
        console.error('Failed to fetch comments:', err);
      } finally {
        setLoading(false);
      }
    };

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
      const data = await apiFetch(`/movies/${movieId}/comments`, {
        method: 'POST',
        body: JSON.stringify({ content: newCommentText.trim() })
      });
      setComments([data.comment, ...comments]);
      setNewCommentText('');
    } catch (err) {
      alert(err.message || 'Failed to post comment.');
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
      const data = await apiFetch(`/movies/${movieId}/comments`, {
        method: 'POST',
        body: JSON.stringify({ content: replyText.trim(), parent_id: parentId })
      });

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
    } catch (err) {
      alert(err.message || 'Failed to post reply.');
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
      const data = await apiFetch(`/comments/${commentId}/like`, {
        method: 'POST'
      });

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
    } catch (err) {
      console.error('Like toggle error:', err);
    }
  };

  // Handle Delete Comment
  const handleDeleteComment = async (commentId, isReply = false, parentId = null) => {
    if (!window.confirm('Delete this comment? This cannot be undone.')) return;

    try {
      await apiFetch(`/comments/${commentId}`, {
        method: 'DELETE'
      });

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
    } catch (err) {
      alert(err.message || 'Failed to delete comment.');
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
          <MessageSquare className="fb-title-icon" />
          <h3 className="fb-comments-title">Audience Reviews</h3>
          <span className="fb-comments-badge">{totalCommentsCount}</span>
        </div>
        <p className="fb-comments-subtitle">
          Discuss, rate, and share your thoughts on <strong>{movieTitle}</strong>
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
                placeholder="Write a public comment about this movie…"
                value={newCommentText}
                onChange={(e) => setNewCommentText(e.target.value)}
                rows={2}
                disabled={submitting}
              />
              <div className="fb-comment-form-footer">
                <span className="fb-input-hint">Press Post to share your thoughts</span>
                <button
                  type="submit"
                  className="fb-btn-submit"
                  disabled={submitting || !newCommentText.trim()}
                >
                  <Send />
                  <span>{submitting ? 'Posting…' : 'Post'}</span>
                </button>
              </div>
            </form>
          </div>
        ) : (
          <div className="fb-auth-prompt">
            <div className="fb-auth-prompt-info">
              <div className="fb-prompt-avatar-placeholder">
                <User />
              </div>
              <span>Sign in to join the discussion and share your thoughts.</span>
            </div>
            <button className="fb-btn-login-prompt" onClick={openAuthModal}>
              <LogIn />
              <span>Sign In</span>
            </button>
          </div>
        )}
      </div>

      {/* Comments List */}
      <div className="fb-comments-list">
        {loading ? (
          <div className="fb-comments-loading">
            <div className="spinner" />
            <span>Loading comments…</span>
          </div>
        ) : comments.length === 0 ? (
          <div className="fb-comments-empty">
            <MessageSquare className="fb-empty-icon" />
            <p>No comments yet</p>
            <p className="fb-empty-hint">Be the first to share your thoughts about this movie!</p>
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
                          <span className="fb-badge-author">You</span>
                        )}
                      </div>
                      <p className="fb-comment-text">{comment.content}</p>

                      {/* Floating like counter on bubble */}
                      {comment.likes_count > 0 && (
                        <div className="fb-bubble-like-counter">
                          <ThumbsUp />
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
                        <ThumbsUp fill={comment.is_liked ? 'currentColor' : 'none'} />
                        <span>Thích</span>
                      </button>

                      <button 
                        className="fb-action-btn"
                        onClick={() => {
                          setReplyingToId(isReplying ? null : comment.id);
                          setReplyText('');
                        }}
                      >
                        <Reply />
                        <span>Reply</span>
                      </button>

                      {user && user.id === comment.user_id && (
                        <button 
                          className="fb-action-btn delete"
                          onClick={() => handleDeleteComment(comment.id, false)}
                          title="Delete this comment"
                        >
                          <Trash2 />
                          <span>Delete</span>
                        </button>
                      )}

                      <span className="fb-comment-time">
                        <Clock />
                        {formatRelativeTime(comment.created_at)}
                      </span>
                    </div>

                    {/* Inline Reply Input Form */}
                    {isReplying && (
                      <div className="fb-reply-box animate-fade-in">
                        <CornerDownRight className="fb-reply-arrow" />
                        <div className="fb-reply-input-wrapper">
                          <input
                            type="text"
                            placeholder={`Reply to ${comment.username}…`}
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
                            <Send />
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
                          {areRepliesVisible ? <ChevronUp /> : <ChevronDown />}
                          <span>
                            {areRepliesVisible 
                              ? `Hide ${comment.replies.length} ${comment.replies.length === 1 ? 'reply' : 'replies'}` 
                              : `View ${comment.replies.length} ${comment.replies.length === 1 ? 'reply' : 'replies'}`}
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
                                        <span className="fb-badge-author">You</span>
                                      )}
                                    </div>
                                    <p className="fb-comment-text">{reply.content}</p>

                                    {reply.likes_count > 0 && (
                                      <div className="fb-bubble-like-counter">
                                        <ThumbsUp />
                                        <span>{reply.likes_count}</span>
                                      </div>
                                    )}
                                  </div>

                                  <div className="fb-comment-actions">
                                    <button 
                                      className={`fb-action-btn ${reply.is_liked ? 'liked' : ''}`}
                                      onClick={() => handleToggleLike(reply.id, true, comment.id)}
                                    >
                                      <ThumbsUp fill={reply.is_liked ? 'currentColor' : 'none'} />
                                      <span>Thích</span>
                                    </button>

                                    {user && user.id === reply.user_id && (
                                      <button 
                                        className="fb-action-btn delete"
                                        onClick={() => handleDeleteComment(reply.id, true, comment.id)}
                                      >
                                        <Trash2 />
                                        <span>Delete</span>
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
