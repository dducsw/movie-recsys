import os
import sys
import time
import glob
import io
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

# Fix Windows console UTF-8 output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_notebooks():
    pipeline_dir = os.path.abspath(os.path.dirname(__file__))
    notebook_files = sorted(glob.glob(os.path.join(pipeline_dir, "0*.ipynb")))

    if not notebook_files:
        print("Không tìm thấy tệp notebook nào trong:", pipeline_dir)
        return

    print("=" * 60)
    print(f"BẮT ĐẦU CHẠY PIPELINE EVALUATION ({len(notebook_files)} NOTEBOOKS)")
    print("=" * 60)

    total_start_time = time.time()
    results = []

    for nb_path in notebook_files:
        nb_name = os.path.basename(nb_path)
        print(f"\n[RUNNING] {nb_name} ...")
        start_time = time.time()

        try:
            with open(nb_path, 'r', encoding='utf-8') as f:
                nb = nbformat.read(f, as_version=4)

            # Khởi tạo executor với timeout 600 giây mỗi cell
            ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
            
            # Chạy notebook trong thư mục chứa notebook để đọc/ghi file đúng vị trí
            ep.preprocess(nb, {'metadata': {'path': pipeline_dir}})

            # Ghi đè lại notebook đã chứa outputs
            with open(nb_path, 'w', encoding='utf-8') as f:
                nbformat.write(nb, f)

            elapsed_time = time.time() - start_time
            print(f"[SUCCESS] {nb_name} hoàn thành trong {elapsed_time:.2f}s")
            results.append((nb_name, "SUCCESS", elapsed_time, None))

        except CellExecutionError as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] Lỗi khi thực thi cell trong {nb_name}!")
            print("-" * 50)
            print(e.ename, ":", e.evalue)
            print("-" * 50)
            results.append((nb_name, "FAILED", elapsed_time, f"{e.ename}: {e.evalue}"))
            # Dừng pipeline nếu notebook trước đó thất bại
            break
        except Exception as e:
            elapsed_time = time.time() - start_time
            print(f"[ERROR] Lỗi không xác định ở {nb_name}: {str(e)}")
            results.append((nb_name, "ERROR", elapsed_time, str(e)))
            break

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 60)
    print("KẾT QUẢ THEO DÕI THỰC THI PIPELINE")
    print("=" * 60)
    print(f"{'Notebook':<35} | {'Trạng thái':<10} | {'Thời gian':<10}")
    print("-" * 60)
    for name, status, duration, err in results:
        print(f"{name:<35} | {status:<10} | {duration:.2f}s")

    print("-" * 60)
    print(f"Tổng thời gian thực thi: {total_elapsed:.2f}s")
    print("=" * 60)

if __name__ == "__main__":
    run_notebooks()
