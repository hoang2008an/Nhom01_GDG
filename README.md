# Nhom01_GDG

Nhóm 1 nhận làm chủ đề **A1. Phát hiện gian lận: Imbalanced dataset**

## Đề bài

### A1. Phát hiện gian lận: Imbalanced dataset

| Hạng mục | Nội dung |
| --- | --- |
| **Câu hỏi** | Khi giao dịch gian lận chỉ chiếm <1% dữ liệu, làm sao để model không "lười" dự đoán tất cả là bình thường? |
| **Gợi ý dữ liệu** | Credit Card Fraud Detection (Kaggle), PaySim, hoặc IEEE-CIS Fraud Detection |
| **Gợi ý hướng thử nghiệm** | - Baseline: Logistic Regression trên dữ liệu gốc<br>- Resampling: undersampling, oversampling, SMOTE, SMOTE-Tomek<br>- Class weight / cost-sensitive learning (`scale_pos_weight`)<br>- Các model khác nhau (Random Forest, XGBoost, LightGBM, CatBoost)<br>- Điều chỉnh ngưỡng quyết định (cost-sensitive threshold tuning)<br>- Multi-model soft voting ensemble |
| **Thang đo** | Precision, Recall, F1, PR-AUC (giải thích vì sao Accuracy không dùng được); Confusion matrix |
| **Bắt buộc** | Chứng minh bằng số liệu vì sao Accuracy gây hiểu lầm; ít nhất 3 cách xử lý mất cân bằng; giải thích đánh đổi Precision và Recall dưới góc nhìn kinh doanh (bỏ sót gian lận và chặn nhầm khách hàng, cái nào tốn kém hơn?) |
| **Mở rộng** | Ước tính thiệt hại/tiết kiệm bằng tiền theo từng ngưỡng; giải thích model bằng SHAP |
| **Tham khảo** | Tài liệu thư viện imbalanced-learn; Chawla et al. (2002), SMOTE |

---

## 🛠 Hướng dẫn cài đặt môi trường & Sửa các lỗi thường gặp

### 1. Khởi tạo môi trường ảo (Khuyến nghị)
Để tránh xung đột package hệ thống, nên tạo virtual environment riêng:
```bash
python -m venv .venv
```
Kích hoạt môi trường:
- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **Windows (CMD):** `.\.venv\Scripts\activate.bat`
- **Linux / macOS:** `source .venv/bin/activate`

---

### 2. Cài đặt các thư viện (Packages)

#### Cách 1: Sử dụng pip (Khuyến nghị)
```bash
pip install -r requirements.txt
```

#### ⚠️ Các lỗi thường gặp khi cài đặt và cách khắc phục:

1. **Lỗi `Microsoft Visual C++ 14.0 or greater is required` khi build wheel cho `greenlet` / `daal`:**
   - **Nguyên nhân:** Máy Windows chưa cài C++ Build Tools nên pip cố gắng compile mã nguồn C/C++.
   - **Cách sửa:** Ép pip chỉ tải bản pre-compiled binary (wheel đã build sẵn):
     ```bash
     pip install --only-binary :all: greenlet
     # hoặc ép tải binary cho tất cả packages:
     pip install --only-binary :all: -r requirements.txt
     ```

2. **Lỗi `ModuleNotFoundError: No module named 'sklearnex'`:**
   - **Nguyên nhân:** File `baseline.ipynb`, `threshold.ipynb` gọi `from sklearnex import patch_sklearn` từ thư viện Intel Extension for Scikit-learn.
   - **Cách sửa:**
     ```bash
     pip install scikit-learn-intelex
     ```

3. **Lỗi Jupyter Kernel Crash khi chạy file `.ipynb`:**
   - **Nguyên nhân:** Thiếu backend `ipykernel` hoặc `ipynb` trong virtual environment.
   - **Cách sửa:**
     ```bash
     pip install ipykernel ipynb
     python -m ipykernel install --user --name=nhom01-gdg-venv --display-name "Python (nhom01-gdg-venv)"
     ```
     *(Sau đó khi mở notebook trong VS Code, bấm chọn Kernel ở góc trên cùng bên phải và chọn **Python (nhom01-gdg-venv)**).*

4. **Lỗi `FileNotFoundError: '../../data/credit_card/creditcard.csv'`:**
   - **Nguyên nhân:** Đường dẫn tương đối `../../data` yêu cầu thư mục `data` nằm ngoài thư mục cha của notebook.
   - **Cách sửa:** Đảm bảo thư mục `data/` chứa dữ liệu csv đã được tải về. Nếu chạy từ notebook con, hãy copy hoặc tạo thư mục `data` ở cấp thư mục ngang hàng dự án hoặc chạy script [kaggle.ipynb](file:///notebooks/kaggle.ipynb).

---

### 3. Tải bộ dữ liệu

Chạy file notebook [notebooks/kaggle.ipynb](file:///notebooks/kaggle.ipynb) hoặc chạy bằng lệnh:
```bash
python -c "import kagglehub; kagglehub.dataset_download('mlg-ulb/creditcardfraud', output_dir='data/credit_card'); kagglehub.dataset_download('ealaxi/paysim1', output_dir='data/paysim')"
```

---

## 📂 Cấu trúc Repository & Danh mục các phương pháp

```text
Nhom01_GDG/
├── data/                                # Chứa các bộ dữ liệu CSV (credit_card, paysim, ieee)
├── notebooks/
│   ├── credit_card/                     # Bộ dữ liệu Credit Card Fraud Detection
│   │   ├── preprocessing.ipynb          # Pipeline nạp và chia tập train/val/test
│   │   ├── baseline.ipynb               # Mô hình cơ sở Logistic Regression
│   │   ├── oversampling.ipynb           # Resampling: RandomOverSampler, SMOTE, ADASYN
│   │   ├── undersampling.ipynb          # Resampling: RandomUnderSampler, ClusterCentroids, NearMiss
│   │   ├── hybrid_resampling.ipynb      # [MỚI] Hybrid: SMOTE-Tomek, SMOTE-ENN
│   │   ├── svm.ipynb                    # Support Vector Machine (SVC)
│   │   ├── threshold.ipynb              # Tinh chỉnh ngưỡng phân lớp (Threshold Tuning)
│   │   ├── tree_based.ipynb             # Tree-based: Random Forest, XGBoost, LightGBM, CatBoost
│   │   ├── hyperparameter.ipynb         # Tối ưu hóa siêu tham số bằng Optuna
│   │   └── optimal_method.ipynb         # [MỚI] PHƯƠNG PHÁP TỐI ƯU: Tri-Model Ensemble + Cost Threshold + SHAP
│   │
│   ├── paysim/                          # Bộ dữ liệu PaySim Mobile Money Fraud
│   │   ├── preprocessing.ipynb          # Tiền xử lý & Feature Engineering (Quy tắc No-Balance leakage)
│   │   ├── 01_baseline.ipynb            # [MỚI] Baseline Logistic Regression & Class Weight
│   │   ├── 02_tree_based.ipynb          # [MỚI] Random Forest, XGBoost, LightGBM với scale_pos_weight
│   │   └── optimal_method.ipynb         # [MỚI] Ensemble + Cost-sensitive Threshold + SHAP
│   │
│   ├── ieee/                            # Bộ dữ liệu IEEE-CIS Fraud Detection
│   │   ├── 01_baseline.ipynb            # Baseline LightGBM & ColumnTransformer
│   │   ├── 02_no_feature_engineering.ipynb
│   │   ├── 03_feature_engineering.ipynb # Feature Engineering sâu (UIDs, frequency, group stats)
│   │   ├── lbgm.ipynb                   # K-Fold LightGBM
│   │   └── optimal_method.ipynb         # [MỚI] Optimal Ensemble trên đặc trưng đa chiều
│   └── kaggle.ipynb                     # Tải dữ liệu từ Kaggle Hub
│
├── BAO_CAO_THUC_NGHIEM_FRAUD_DETECTION.md # Báo cáo phân tích chuyên sâu chi tiết (Toàn bộ 11 phương pháp)
├── run_all_benchmarks.py                # Script chạy tự động đánh giá 11 phương pháp trên 3 bộ dữ liệu
└── requirements.txt                     # Danh sách thư viện phụ thuộc
```

---

## 🏆 Tóm tắt kết quả & Phương pháp tối ưu nhất (Optimal Method)

### 1. Vì sao Accuracy gây hiểu lầm? (Accuracy Paradox)
Trên tập Credit Card ($85,443$ mẫu test, $148$ ca gian lận):
- Một mô hình ngây thơ chỉ đoán toàn bộ là bình thường đạt **Accuracy = 99.827%**.
- Nhưng **Recall = 0.00%**, bỏ sót $100\%$ các vụ gian lận, gây thiệt hại ước tính **\$29,600** trong một đợt kiểm tra.
- **Thang đo bắt buộc:** **PR-AUC (Precision-Recall AUC)**, **Recall**, **Precision** và **F1-Score**.

### 2. Phương pháp tối ưu toàn diện nhất
Mô hình **Tri-Model Soft Voting Ensemble (XGBoost + LightGBM + CatBoost) kết hợp Cost-Sensitive Threshold Tuning**:
- Sử dụng trọng số phạt `scale_pos_weight` thay vì resampling đơn thuần (tránh làm mất cân đối phân phối gốc).
- Hòa trộn xác suất 3 kiến trúc: $P_{\text{ensemble}} = 0.4 \cdot P_{\text{XGB}} + 0.4 \cdot P_{\text{LGB}} + 0.2 \cdot P_{\text{CatBoost}}$.
- Tối ưu hóa ngưỡng quyết định theo chi phí thực tế: $\mathcal{L} = 200 \times \text{FN} + 10 \times \text{FP}$.
- **Hiệu quả:** Đạt **PR-AUC cao nhất trên cả 3 bộ dữ liệu** ($0.8166$ Credit Card, $0.9909$ PaySim, $0.8133$ IEEE-CIS) và **tiết kiệm hơn 81% chi phí tổn thất** so với mô hình cơ sở.
