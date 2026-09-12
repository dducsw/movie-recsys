# 🎨 MovieNex Frontend (React + Vite + Tailwind CSS)

Frontend của **MovieNex** là giao diện người dùng theo phong cách nền tảng streaming hiện đại (lấy cảm hứng từ Netflix và Prime Video), được xây dựng bằng **React 18**, **Vite** và **Tailwind CSS**. Ứng dụng mang lại trải nghiệm mượt mà, phản hồi thời gian thực và tích hợp trực tiếp với bộ máy gợi ý 3 Lớp cùng Trợ lý ảo AI đàm thoại.

---

## 📸 Giao diện Điển hình

![MovieNex Interface](../../assest/image.png)

---

## ✨ Tính năng Nổi bật

* **Giao diện Streaming Hiện đại (Cinema Grade UI)**:
  * Hiệu ứng Dark/Light Mode tùy biến.
  * Hero Banner động làm nổi bật các bộ phim bom tấn với trailer và thông tin tổng quan.
  * Các hàng phim (Movie Rows) trượt mượt mà theo từng danh mục: *Trending Now*, *Dành riêng cho bạn (Personalized For You)*, *Top Đánh Giá*, *Hành động*, *Viễn tưởng*, v.v.
* **Cá nhân hóa & Khảo sát Người dùng Mới (Onboarding Flow)**:
  * Modal khảo sát sở thích thể loại (`OnboardingModal`) xuất hiện ngay sau khi đăng ký, giúp giải quyết triệt để vấn đề khởi đầu lạnh (Cold Start).
* **Trợ lý AI Đàm thoại Thông minh (`ChatbotPage` & `ChatbotView`)**:
  * Trò chuyện tự nhiên bằng Tiếng Việt hoặc Tiếng Anh.
  * Tìm kiếm phim theo cốt truyện, cảm xúc, diễn viên hoặc đạo diễn.
  * Hiển thị danh sách thẻ phim gợi ý tương tác trực tiếp ngay trong khung chat.
* **Chi tiết Phim & Đánh giá Tương tác (`MovieDetailPage`)**:
  * Xem thông tin đầy đủ, danh sách diễn viên, đạo diễn.
  * Phần bình luận và chấm điểm sao (`CommentSection`).
  * Danh sách phim tương tự (`Similar Movies`) được tính toán từ vector embedding của Qdrant.
* **Danh sách Yêu thích (`WatchlistPage`)**:
  * Lưu trữ và quản lý các bộ phim muốn xem sau.
* **Xác thực Người dùng (`AuthModal`)**:
  * Đăng ký, đăng nhập tài khoản an toàn với JWT Bearer Token được quản lý qua `AuthContext`.

---

## 🗂️ Cấu trúc Thư mục

```
web_app/frontend/
├── index.html              # HTML Entrypoint
├── package.json            # Thư viện npm phụ thuộc
├── vite.config.js          # Cấu hình Vite dev server & proxy
├── nginx.conf              # Cấu hình web server Nginx khi chạy trong Docker
├── Dockerfile              # Multi-stage Dockerfile tối ưu kích thước
├── public/                 # Tài nguyên tĩnh (favicon, icons)
└── src/
    ├── main.jsx            # React root mount
    ├── App.jsx             # Định tuyến URL (React Router) & Layout chính
    ├── App.css             # Stylesheet tổng thể
    ├── index.css           # Cấu hình Tailwind CSS & CSS variables
    ├── api/                # Các module gọi REST API tới FastAPI Backend
    │   ├── client.js       # Cấu hình Axios Instance & interceptor JWT
    │   ├── auth.js         # API login, register, me, watchlist
    │   ├── movies.js       # API danh mục phim, chi tiết, trending
    │   ├── recsys.js       # API gợi ý 3-Stage (For You, Similar)
    │   └── chatbot.js      # API gửi tin nhắn, lịch sử chat
    ├── context/            # Quản lý State toàn cục
    │   ├── AuthContext.jsx       # State người dùng đăng nhập & token
    │   ├── ThemeContext.jsx      # Chuyển đổi Dark / Light Theme
    │   └── WatchlistContext.jsx  # Quản lý danh sách phim đã lưu
    ├── components/         # Các UI Components tái sử dụng
    │   ├── TopNav/         # Thanh điều hướng trên cùng kèm thanh tìm kiếm
    │   ├── HeroBanner/     # Banner spotlight phim nổi bật
    │   ├── MovieCard/      # Thẻ phim với poster, điểm vote, hover quick actions
    │   ├── MovieRow/       # Hàng phim cuộn ngang theo thể loại
    │   ├── ChatbotView/    # Drawer / Khung chat đàm thoại AI
    │   ├── CommentSection/ # Bình luận và đánh giá sao người dùng
    │   ├── GenreFilterBar/ # Thanh lọc phim theo thể loại dạng pills
    │   ├── AuthModal.jsx   # Dialog Đăng nhập / Đăng ký
    │   └── OnboardingModal.jsx # Khảo sát chọn thể loại cho user mới
    └── pages/              # Các trang chính
        ├── HomePage.jsx        # Trang chủ Streaming
        ├── ExplorePage.jsx     # Trang khám phá & lọc kho phim
        ├── MovieDetailPage.jsx # Trang chi tiết phim & đánh giá
        ├── WatchlistPage.jsx   # Trang phim yêu thích
        └── ChatbotPage.jsx     # Trang Trợ lý ảo AI toàn màn hình
```

---

## 🚀 Hướng dẫn Khởi chạy & Phát triển

### 1. Cài đặt Dependencies
Đảm bảo bạn đã cài đặt **Node.js 18+**:

```bash
cd web_app/frontend
npm install
```

### 2. Cấu hình Môi trường
Tạo file `.env` tại `web_app/frontend/.env` (nếu cần đổi địa chỉ backend):

```env
VITE_API_URL=http://localhost:8000
```
*(Mặc định frontend kết nối tới backend tại cổng `8000`)*.

### 3. Khởi chạy Chế độ Development (Vite HMR)
```bash
npm run dev
```
Ứng dụng sẽ chạy tại: **[http://localhost:5173](http://localhost:5173)**.

### 4. Build Production Bundle
```bash
npm run build
```
Thư mục xuất bản tối ưu hóa sẽ nằm tại `web_app/frontend/dist/`.

Để xem thử bản build production:
```bash
npm run preview
```

---

## 🐳 Triển khai qua Docker
Frontend đã được cấu hình với Multi-stage Docker build kết hợp Nginx để phục vụ file tĩnh:

```bash
docker build -t movienex-frontend .
docker run -p 5173:80 movienex-frontend
```
Hoặc khởi chạy thông qua `docker compose --profile app up -d frontend`.
