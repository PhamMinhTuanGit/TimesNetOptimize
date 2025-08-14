from neuralforecast.models import NHITS
from neuralforecast.losses.pytorch import MAE, DistributionLoss

# Các tham số chính để tạo mô hình lớn
large_mlp_units = [768, 768]
num_blocks_per_stack = 3

model = NHITS(
    h=24,
    input_size=72,
    # Tăng đáng kể kích thước các lớp MLP
    mlp_units=[large_mlp_units] * 3,
    # Sử dụng nhiều khối trong mỗi stack
    n_blocks=[num_blocks_per_stack] * 3,
    # Giữ các stack ở dạng 'identity' để có nhiều tham số nhất
    stack_types=['identity', 'identity', 'identity'],
    # Các tham số khác có thể giữ nguyên hoặc điều chỉnh
    learning_rate=1e-4, # Cân nhắc giảm learning rate cho mô hình lớn
    max_steps=1000,
    loss=DistributionLoss(distribution='Normal', level=[80, 90]),
    val_check_steps=50,
)

from model import count_parameters
print(f"Số lượng tham số của mô hình N-HITS: {count_parameters(model):,}")
# Kết quả sẽ xấp xỉ 21-22 triệu tham số