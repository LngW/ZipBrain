import onnxruntime as ort
import numpy as np

# 1. 设置 Provider 配置
# 优先使用 TensorrtExecutionProvider，如果失败则回退到 CUDA 或 CPU
providers = [
    # ('TensorrtExecutionProvider', {
    #     'device_id': 0,
    #     'trt_fp16_enable': True,  # 开启 FP16 加速
    #     'trt_max_workspace_size': 1 << 30, # 1GB 显存占用
    # }),
    ('CUDAExecutionProvider', {
        'device_id': 0,
    }),
    'CPUExecutionProvider'
]

# 创建 SessionOptions 实例
options = ort.SessionOptions()

# 关键设置：禁用线程亲和力设置
options.add_session_config_entry("session.use_per_session_threads", "1")
options.intra_op_num_threads = 4
options.enable_profiling = True

# 2. 加载模型
# 第一次加载包含 STFT 的模型会比较慢，因为 ORT 会在后台构建编译 Engine
session = ort.InferenceSession("model.onnx", sess_options=options, providers=providers)

# 3. 准备数据 (假设你的 EEG 输入是 1D 信号)
# 请根据你模型的实际 input_shape 调整
for _ in range(10):
    dummy_input = np.random.randn(1, 16, 1000).astype(np.float32) 
    input_name = session.get_inputs()[0].name

    # 4. 执行推理
    outputs = session.run(None, {input_name: dummy_input})

    # 5. 后处理 (之前讨论的 6 分类逻辑)
    logits = outputs[0]
    predicted_class = np.argmax(logits)
    print(f"预测类别索引: {predicted_class}")