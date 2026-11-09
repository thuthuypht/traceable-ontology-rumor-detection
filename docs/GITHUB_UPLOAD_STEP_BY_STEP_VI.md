# Từng bước tải mã nguồn lên GitHub (Windows, sử dụng trình duyệt)

## Phân biệt hai thư mục sau khi giải nén
- **GITHUB_UPLOAD_THIS_FOLDER:** mã nghiên cứu và các kết quả tổng hợp được chuẩn bị để upload.
- **PRIVATE_RESEARCH_ARCHIVE_DO_NOT_UPLOAD:** kết quả đầy đủ, mô hình, bộ tiền xử lý, notebook nhúng dữ liệu và bản thảo Word. **Không tải thư mục này lên GitHub công khai.**

## Cách A — trình duyệt, đơn giản nhất
1. Tải ZIP trong ChatGPT về máy, bấm chuột phải > Extract All (Giải nén tất cả).
2. Đăng nhập https://github.com/thuthuypht rồi vào https://github.com/new.
3. Owner: **thuthuypht**; Repository name đề xuất: **traceable-ontology-rumor-detection**; description: `Traceable fuzzy structural evidence, reproducible rule verification, model sensitivity, and cross-domain rumor detection audits`.
4. Đặt **Private** ở giai đoạn rà soát; không tích chọn Add README, .gitignore hay license vì gói đã có README và .gitignore; nhấn Create repository.
5. Vào repository vừa tạo > Add file > Upload files (hoặc liên kết `uploading` khi repo trống).
6. Mở thư mục **GITHUB_UPLOAD_THIS_FOLDER** đã giải nén. Chọn các *file và thư mục bên trong*, không chọn thư mục cha, kéo thả vào GitHub; nếu trình duyệt không giữ cây thư mục, dùng GitHub Desktop theo cách B. Giới hạn giao diện GitHub: tối đa 100 file một lần, 25 MiB mỗi file.
7. Kiểm tra đường dẫn root sẽ có `README.md`, `src/`, `results/`, `paper/`, `docs/`, `scripts/`, `notebooks/`, `.gitignore`. Không có `PRIVATE_RESEARCH_ARCHIVE_DO_NOT_UPLOAD`.
8. Nhập commit message `Initial research release: source code and verified aggregate results`, chọn Commit changes. Kiểm tra README hiển thị ở trang chính.
9. Vào `scripts/verify_paper_tables.py` và các CSV; trước khi đặt Public phải kiểm tra quyền công bố dữ liệu, quyền tác giả, tài trợ và chọn LICENSE. Khi sẵn sàng, vào Settings > General > Danger Zone > Change repository visibility.

## Cách B — GitHub Desktop, an toàn cho nhiều thư mục
1. Cài và đăng nhập https://desktop.github.com/.
2. Sau khi tạo repo trống, ở GitHub Desktop chọn File > Clone repository > URL, dùng `https://github.com/thuthuypht/traceable-ontology-rumor-detection.git`, chọn thư mục trên máy.
3. Sao chép **nội dung bên trong** `GITHUB_UPLOAD_THIS_FOLDER` vào thư mục repo đã clone (không chép thư mục PRIVATE).
4. Quay lại GitHub Desktop kiểm tra Files changed, đảm bảo không có dữ liệu tweet hoặc private checkpoints.
5. Gõ Summary `Initial research release: sources and aggregate outputs`, chọn Commit to main rồi chọn Push origin.
6. Mở GitHub kiểm tra cây thư mục và README.

## Cập nhật những lần sau
- Thay hoặc thêm file trong bản clone, **không sửa chồng** thư mục private vào repo.
- Commit với message mô tả rõ phiên bản nghiên cứu, Push origin.
- Gắn tag như `v0.1-research-audit` sau khi kiểm thử và kiểm tra chính sách dữ liệu.

## Kiểm thử trước khi public
```powershell
python scripts/verify_paper_tables.py
python scripts/validate_reproducible_run.py
python scripts/verify_release_sha256.py
```

**Quan trọng:** Tên repo nói trên là *đề xuất*; repository chưa được tạo tự động. Nếu đã có repo phù hợp khác, thay tên tương ứng trong URL.
