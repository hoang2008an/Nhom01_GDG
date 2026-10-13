# Nhom01_GDG

Nhóm 1 nhận làm chủ đề **A1. Phát hiện gian lận: Imbalanced dataset**


Đề bài

### A1. Phát hiện gian lận: Imbalanced dataset

| Hạng mục                             | Nội dung                                                                                                                                                                                                                                                         |
| -------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Câu hỏi**                    | Khi giao dịch gian lận chỉ chiếm <1% dữ liệu, làm sao để model không "lười" dự đoán tất cả là bình thường?                                                                                                                                   |
| **Gợi ý dữ liệu**            | Credit Card Fraud Detection (Kaggle), PaySim, hoặc IEEE-CIS Fraud Detection                                                                                                                                                                                      |
| **Gợi ý hướng thử nghiệm** | - Baseline: Logistic Regression trên dữ liệu gốc- Resampling: undersampling, oversampling, SMOTE- Class weight / cost-sensitive learning- Các model khác nhau (Random Forest, XGBoost...)- Điều chỉnh ngưỡng quyết định (threshold)                 |
| **Thang đo**                    | Precision, Recall, F1, PR-AUC (giải thích vì sao Accuracy không dùng được); Confusion matrix                                                                                                                                                              |
| **Bắt buộc**                   | Chứng minh bằng số liệu vì sao Accuracy gây hiểu lầm; ít nhất 3 cách xử lý mất cân bằng; giải thích đánh đổi Precision và Recall dưới góc nhìn kinh doanh (bỏ sót gian lận và chặn nhầm khách hàng, cái nào tốn kém hơn?) |
| **Mở rộng**                    | Ước tính thiệt hại/tiết kiệm bằng tiền theo từng ngưỡng; giải thích model bằng SHAP                                                                                                                                                                |
| **Tham khảo**                   | Tài liệu thư viện imbalanced-learn; Chawla et al. (2002), SMOTE                                                                                                                                                                                               |

### Hướng dẫn sử dụng repo

**Tải package:**

**Cách 1:** sử dụng pip

```
pip install -r requirements.txt
```

**Cách 2:** sử dụng uv

```
uv sync
```

**Lấy data:**

```
chạy notebooks\kaggle.ipynb
```
