# TRƯỜNG ĐẠI HỌC PHENIKAA  
# KHOA CÔNG NGHỆ THÔNG TIN

## BÁO CÁO ĐỒ ÁN CƠ SỞ  
**<Team ID>**  
**Dự án:** Dự báo khách hàng rời bỏ dịch vụ viễn thông (Telco Customer Churn Prediction)

**Danh sách sinh viên:**  
- <Họ và tên 1> - <MSSV> - <email>  
- <Họ và tên 2> - <MSSV> - <email>  
- <Họ và tên 3> - <MSSV> - <email>

**Giảng viên hướng dẫn:** <Supervisor>  
**Ngày:** <date>

**GitHub dự án:** https://github.com/<your-org-or-user>/<your-repo>

---

## 1. Giới thiệu

### 1.1 Đặt vấn đề
Bài toán rời bỏ dịch vụ (customer churn) là vấn đề trọng yếu trong các doanh nghiệp viễn thông. Việc mất khách hàng hiện hữu làm giảm doanh thu định kỳ, tăng chi phí tìm kiếm khách hàng mới và ảnh hưởng trực tiếp đến năng lực cạnh tranh. Dự án này xây dựng mô hình học máy nhằm dự báo khả năng rời bỏ dịch vụ của từng khách hàng dựa trên dữ liệu hành vi và hợp đồng.

**Định nghĩa bài toán:**
- **Input (đầu vào):** thông tin nhân khẩu học và hành vi sử dụng dịch vụ của khách hàng, gồm: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
- **Output (đầu ra):** nhãn phân loại `Churn` (Yes/No), tương ứng khả năng khách hàng rời bỏ dịch vụ.

**Ý nghĩa bài toán:**
- **Đối với cá nhân:** khách hàng có nguy cơ rời bỏ có thể được tư vấn gói cước phù hợp hơn.
- **Đối với tổ chức:** doanh nghiệp tối ưu chiến dịch giữ chân khách hàng (retention), giảm chi phí marketing không cần thiết.
- **Đối với xã hội (địa phương và toàn cầu):** nâng cao chất lượng dịch vụ số, tối ưu sử dụng hạ tầng viễn thông, thúc đẩy chuyển đổi số bền vững.

### 1.2 Các giải pháp đã có
Hiện nay doanh nghiệp thường dùng ba hướng chính:
1. **Rule-based/Heuristic:** xây luật thủ công theo hợp đồng, phí dịch vụ, tần suất sử dụng.
2. **Phân tích thống kê truyền thống:** dùng kiểm định, mô tả tương quan để đưa ra ngưỡng cảnh báo.
3. **Machine Learning:** huấn luyện mô hình từ dữ liệu lịch sử để tự học mẫu churn.

**Hạn chế của các giải pháp phổ biến:**
- Rule-based phụ thuộc kinh nghiệm chuyên gia, khó mở rộng và khó cập nhật.
- Thống kê mô tả thiếu năng lực dự báo ở mức từng khách hàng.
- Nhiều nghiên cứu ML chỉ dừng ở notebook, thiếu tích hợp ứng dụng để sử dụng thực tế.

### 1.3 Giải pháp đề xuất
Nhóm đề xuất pipeline hoàn chỉnh gồm:
1. Tiền xử lý dữ liệu Telco churn dạng bảng.
2. Huấn luyện và so sánh tối thiểu 3 mô hình học máy: Logistic Regression, Random Forest, SVM.
3. Chọn mô hình tốt nhất theo chỉ số F1 trên tập kiểm thử.
4. Tích hợp mô hình tốt nhất vào ứng dụng web đơn giản bằng Streamlit để dự báo theo dữ liệu nhập tay.

---

## 2. Thiết kế và triển khai

### 2.1 Các yêu cầu chức năng
Hệ thống cần đáp ứng các chức năng sau:
1. Đọc dữ liệu từ file CSV chuẩn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`).
2. Tiền xử lý dữ liệu tự động (chuyển kiểu, xử lý giá trị thiếu, mã hóa biến phân loại).
3. Huấn luyện 3 mô hình ML và tối ưu siêu tham số bằng GridSearchCV.
4. Đánh giá mô hình bằng các chỉ số: Accuracy, Precision, Recall, F1, ROC-AUC.
5. Xuất bảng so sánh mô hình và lưu model tốt nhất.
6. Ứng dụng web cho phép nhập thông tin khách hàng và trả ra:
   - Dự đoán Churn Yes/No
   - Xác suất churn.

### 2.2 Các yêu cầu phi chức năng
- **Tái lập (reproducibility):** dùng `random_state=42`, script train chạy một lệnh.
- **Hiệu năng:** thời gian huấn luyện phù hợp máy cá nhân ở quy mô dữ liệu ~7k bản ghi.
- **Khả dụng:** giao diện web đơn giản, dễ sử dụng với người không chuyên ML.
- **Bảo trì:** tách module `src/churn/training.py` và `scripts/train_models.py`.

### 2.3 Các ràng buộc (Constraints)
- Dữ liệu công khai, không chứa thông tin định danh cá nhân thực (PII) ở mức nhạy cảm.
- Phạm vi đồ án đại học: ưu tiên mô hình có khả năng giải thích và triển khai nhanh.
- Thời gian triển khai ngắn nên tập trung vào bài toán tabular classification.

### 2.4 Các ràng buộc về triển khai

#### 2.4.1 Các ràng buộc kinh tế
- Sử dụng hoàn toàn mã nguồn mở: Python, Scikit-learn, Streamlit.
- Không cần chi phí hạ tầng cloud bắt buộc trong giai đoạn đồ án.
- Có thể chạy trên laptop cá nhân (CPU) mà không cần GPU.

#### 2.4.2 Các ràng buộc về đạo đức
- Tránh dùng mô hình để từ chối dịch vụ tự động gây bất lợi cho nhóm khách hàng yếu thế.
- Cần minh bạch mục đích: mô hình là công cụ hỗ trợ quyết định, không thay thế hoàn toàn con người.
- Cần giám sát sai lệch theo nhóm (ví dụ theo độ tuổi/giới tính) khi triển khai thực tế.

### 2.5 Mô hình hệ thống / Thiết kế giải pháp

#### 2.5.1 Các kịch bản của hệ thống (Use-cases)
- **UC1 - Data Scientist:** chạy script train, so sánh mô hình, lưu model tốt nhất.
- **UC2 - Nhân sự chăm sóc khách hàng:** nhập thông tin khách hàng trên web và nhận dự báo churn.
- **UC3 - Quản lý kinh doanh:** xem chỉ số mô hình và đề xuất chiến lược giữ chân khách hàng.

#### 2.5.2 Mô hình Use-case
Tác nhân chính gồm `Data Scientist`, `Business User`.  
Hệ thống gồm 2 phân hệ: `Training Pipeline` và `Web Inference App`.

#### 2.5.3 Mô hình lớp và đối tượng
- `training.py`:
  - `load_dataset()`
  - `split_features_target()`
  - `build_preprocessor()`
  - `train_and_select_best()`
- `train_models.py`: điều phối toàn bộ tiến trình train/evaluate/save.
- `app.py`: tải model `best_model.joblib`, dựng form nhập liệu, suy luận và hiển thị kết quả.

#### 2.5.4 Các biểu đồ tuần tự (mô tả)
**Luồng huấn luyện:**
1. Đọc CSV dữ liệu.
2. Tiền xử lý (impute + one-hot + scale).
3. GridSearchCV cho từng mô hình.
4. Đánh giá trên tập test.
5. Chọn model có F1 cao nhất.
6. Lưu artifacts (`model_comparison.csv`, `best_model.joblib`, `best_model_summary.json`).

**Luồng suy luận web:**
1. Người dùng nhập dữ liệu khách hàng.
2. App đóng gói payload thành DataFrame.
3. Model trả về nhãn dự đoán + xác suất churn.
4. App hiển thị kết quả cho người dùng.

#### 2.5.5 Các màn hình giao diện người dùng
- Màn hình chính:
  - Form nhập các thuộc tính khách hàng.
  - Nút “Dự báo Churn”.
- Màn hình kết quả:
  - Nhãn Churn Yes/No.
  - Thanh xác suất churn.
  - Bảng dữ liệu đầu vào đã dùng để dự báo.

---

## 3. Một số thành phần khác của đồ án

### 3.1 Kế hoạch dự án

#### 3.1.1 Kế hoạch theo giai đoạn
- **Tuần 1:** xác định bài toán, khảo sát dữ liệu, EDA cơ bản.
- **Tuần 2:** xây pipeline tiền xử lý, huấn luyện 3 mô hình.
- **Tuần 3:** đánh giá, chọn mô hình tốt nhất, lưu artifacts.
- **Tuần 4:** tích hợp web app, hoàn thiện báo cáo và đóng gói nộp bài.

#### 3.1.2 Phân công công việc
| Thành viên | Vai trò chính | Công việc |
|---|---|---|
| SV1 | Data & Modeling | Tiền xử lý dữ liệu, Logistic Regression, tổng hợp metric |
| SV2 | Modeling & Evaluation | Random Forest, SVM, tuning và lựa chọn model |
| SV3 | Deployment & Report | Streamlit app, tài liệu, tổng hợp báo cáo |

#### 3.1.3 Gantt Chart (dạng bảng)
| Công việc | T1 | T2 | T3 | T4 |
|---|---|---|---|---|
| Thu thập & phân tích dữ liệu | ███ |  |  |  |
| Xây dựng pipeline train |  | ███ |  |  |
| Thử nghiệm và chọn model |  | ███ | ███ |  |
| Tích hợp web app |  |  | ███ | █ |
| Viết báo cáo & đóng gói nộp |  |  | █ | ███ |

### 3.2 Đảm bảo thực hiện đúng làm việc nhóm
- Quản lý mã nguồn bằng Git/GitHub.
- Mỗi thành viên phụ trách module rõ ràng, có review chéo.
- Họp nhóm định kỳ để cập nhật tiến độ và xử lý rủi ro.

### 3.3 Các vấn đề về đạo đức và làm việc chuyên nghiệp
- Tôn trọng bản quyền dữ liệu và ghi nguồn đầy đủ.
- Không sử dụng dữ liệu cá nhân nhạy cảm.
- Minh bạch phương pháp đánh giá, không “chọn lọc” metric có lợi.

### 3.4 Tác động xã hội
- **Tích cực:** nâng cao chất lượng dịch vụ, giảm churn, tối ưu chi phí vận hành.
- **Rủi ro:** mô hình có thể sai lệch nếu dữ liệu không đại diện theo thời gian hoặc theo nhóm người dùng.
- **Giảm thiểu:** tái huấn luyện định kỳ, theo dõi drift dữ liệu, kiểm tra fairness.

### 3.5 Kế hoạch cho kiến thức mới và chiến lược học tập
- Tự học mở rộng về Explainable AI (SHAP/LIME) để giải thích dự đoán.
- Học về MLOps cơ bản (versioning model, theo dõi thí nghiệm).
- Nâng cấp app từ bản demo thành dashboard theo dõi churn theo phân khúc.

---

## 4. Kết luận
Nhóm đã hoàn thành vòng đời bài toán ML ở mức đồ án đại học:
1. Xác định rõ bài toán churn prediction cùng input/output và ý nghĩa ứng dụng.
2. Sử dụng dataset công khai Telco Churn để huấn luyện/kiểm thử.
3. Trình bày và triển khai 3 phương pháp ML: Logistic Regression, Random Forest, SVM.
4. Thực nghiệm và chọn mô hình tốt nhất theo F1 trên tập test.
5. Thảo luận kết quả và các rủi ro khi triển khai.
6. Tích hợp mô hình tốt nhất vào ứng dụng web đơn giản.

**Kết quả thực nghiệm (dữ liệu hiện tại):**
- Dataset: 7,043 mẫu, 21 cột.
- Tỷ lệ nhãn: Churn=No (5,174), Churn=Yes (1,869).
- Mô hình tốt nhất: **Random Forest**.

**Bảng so sánh mô hình (trích xuất từ chạy thực tế):**

| Model | CV Best F1 | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.6345 | 0.7644 | 0.5418 | 0.7273 | **0.6210** | 0.8404 |
| Logistic Regression | 0.6333 | 0.7410 | 0.5078 | 0.7834 | 0.6162 | **0.8408** |
| SVM | 0.6215 | 0.7438 | 0.5115 | 0.7727 | 0.6155 | 0.8211 |

Nhìn chung, Random Forest cho F1 cao nhất, phù hợp mục tiêu cân bằng giữa phát hiện churn và hạn chế cảnh báo sai.

---

## 5. Tài liệu tham khảo
[1] Kaggle, “Telco Customer Churn Dataset”, https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
[2] Scikit-learn Documentation, https://scikit-learn.org/stable/  
[3] Streamlit Documentation, https://docs.streamlit.io/  
[4] IBM, Telco Churn sample data and resources, https://github.com/IBM/telco-customer-churn-on-icp4d

---

## Phụ lục A - Đối chiếu yêu cầu bài tập lớn

| Yêu cầu môn học | Trạng thái | Minh chứng |
|---|---|---|
| 1) Trình bày bài toán, input/output, ý nghĩa | Hoàn thành | Mục 1.1 |
| 2) Xây dựng tập dữ liệu huấn luyện/kiểm thử | Hoàn thành | Dữ liệu Telco Churn, split train/test trong pipeline |
| 3) Trình bày tối thiểu 3 phương pháp ML | Hoàn thành | Logistic Regression, Random Forest, SVM |
| 4) Thử nghiệm chọn mô hình tốt nhất | Hoàn thành | `reports/model_comparison.csv`, chọn theo Test F1 |
| 5) Thảo luận kết quả | Hoàn thành | Mục 4 và các mục ràng buộc/tác động |
| 6) Tích hợp mô hình tốt nhất vào web app | Hoàn thành | `app/app.py`, `models/best_model.joblib` |

---

## Phụ lục B - Bảng đánh giá đóng góp thành viên (mẫu)

| STT | Họ và tên | MSSV | Vai trò chính | Mức độ đóng góp (%) | Ghi chú |
|---|---|---|---|---:|---|
| 1 | <SV1> | <MSSV> | Data preprocessing + modeling | 34 |  |
| 2 | <SV2> | <MSSV> | Model tuning + evaluation | 33 |  |
| 3 | <SV3> | <MSSV> | Web app + report | 33 |  |

Tổng: 100%
