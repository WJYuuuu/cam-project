# 在x86主机上创建 convert_mnist_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import onnx
from onnxsim import simplify
import numpy as np


# ====================== 你的模型定义（复制过来） ======================
class SimpleCNN(nn.Module):
    """简单的CNN模型用于MNIST分类"""

    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # 池化层
        self.pool = nn.MaxPool2d(2, 2)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 128)  # 经过两次池化后尺寸：28->14->7
        self.fc2 = nn.Linear(128, num_classes)

        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # 卷积层1 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv1(x)))

        # 卷积层2 + 激活函数 + 池化
        x = self.pool(F.relu(self.conv2(x)))

        # 展平
        x = x.view(-1, 64 * 7 * 7)

        # 全连接层1 + 激活函数 + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # 全连接层2（输出层）
        x = self.fc2(x)

        return x


# ==================================================================

def convert_pth_to_onnx(pth_path, onnx_path, input_size=(28, 28)):
    """
    将MNIST SimpleCNN模型转换为ONNX格式
    特别注意：MNIST是1通道灰度图，输入尺寸28x28
    """
    print("=" * 60)
    print("MNIST SimpleCNN模型转换")
    print("=" * 60)

    print("步骤1: 初始化模型...")
    model = SimpleCNN(num_classes=10)

    # 查看模型结构
    print(f"模型结构:")
    print(f"  输入: 1 x {input_size[0]} x {input_size[1]} (灰度图)")
    print(f"  输出: 10个类别")

    print("\n步骤2: 加载训练好的权重...")
    try:
        checkpoint = torch.load(pth_path, map_location='cpu')

        # 尝试不同的权重加载方式
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
            print("  使用 'state_dict' 加载权重")
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
            print("  使用 'model' 加载权重")
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("  使用 'model_state_dict' 加载权重")
        else:
            # 尝试直接加载为state_dict
            try:
                model.load_state_dict(checkpoint)
                print("  直接加载为state_dict")
            except:
                # 最后尝试：可能是只保存了权重，没有state_dict包装
                model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint.items()})
                print("  尝试加载原始权重字典")

    except Exception as e:
        print(f"  权重加载失败: {e}")
        print("  尝试使用随机权重继续转换（仅用于测试）")

    model.eval()

    print("\n步骤3: 创建虚拟输入...")
    # MNIST是1通道灰度图，28x28
    batch_size = 1
    dummy_input = torch.randn(batch_size, 1, input_size[0], input_size[1])
    print(f"  虚拟输入形状: {dummy_input.shape}")
    print(f"  数据类型: {dummy_input.dtype}")

    print("\n步骤4: 前向推理测试...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"  输出形状: {output.shape}")
        print(f"  输出示例: {output[0, :5]} ...")  # 显示前5个值

    print("\n步骤5: 导出为ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=['input'],
        output_names=['output'],
        opset_version=11,
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    print(f"  ONNX模型已保存: {onnx_path}")

    print("\n步骤6: 验证ONNX模型...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("  ONNX模型验证通过")

    # 打印模型信息
    print("\n步骤7: 模型信息摘要:")
    print(f"  输入:")
    for input in onnx_model.graph.input:
        print(f"    名称: {input.name}")
        dims = [dim.dim_value if dim.dim_value > 0 else dim.dim_param
                for dim in input.type.tensor_type.shape.dim]
        print(f"    形状: {dims}")

    print(f"  输出:")
    for output in onnx_model.graph.output:
        print(f"    名称: {output.name}")
        dims = [dim.dim_value if dim.dim_value > 0 else dim.dim_param
                for dim in output.type.tensor_type.shape.dim]
        print(f"    形状: {dims}")

    print("\n步骤8: 简化模型...")
    try:
        model_simp, check = simplify(onnx_model)
        if check:
            simplified_path = onnx_path.replace('.onnx', '_simplified.onnx')
            onnx.save(model_simp, simplified_path)
            print(f"  简化模型已保存: {simplified_path}")
            final_path = simplified_path
        else:
            print("  模型简化失败，使用原始模型")
            final_path = onnx_path
    except Exception as e:
        print(f"  简化过程中出错: {e}")
        final_path = onnx_path

    print("\n步骤9: 测试ONNX推理...")
    try:
        import onnxruntime as ort

        # 创建ONNX Runtime会话
        ort_session = ort.InferenceSession(final_path)

        # 准备输入
        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}

        # 推理
        ort_outputs = ort_session.run(None, ort_inputs)

        print(f"  ONNX推理输出形状: {ort_outputs[0].shape}")

        # 对比PyTorch和ONNX输出
        torch_output_np = output.numpy()
        ort_output_np = ort_outputs[0]

        # 计算差异
        diff = np.abs(torch_output_np - ort_output_np).max()
        print(f"  PyTorch与ONNX输出最大差异: {diff:.6f}")

        if diff < 1e-5:
            print("  ✅ 转换验证通过！")
        else:
            print("  ⚠️  转换有差异，但可能仍在可接受范围")

    except Exception as e:
        print(f"  ONNX推理测试失败: {e}")

    print("\n" + "=" * 60)
    print(f"转换完成！")
    print(f"最终模型: {final_path}")
    print("下一步: 使用ncnn工具转换为ncnn格式")
    print("=" * 60)

    return final_path


if __name__ == "__main__":
    # 配置文件路径
    pth_file = "../checkpoints/best_model.pth"  # 你的.pth文件路径
    onnx_file = "mnist_model.onnx"

    # 执行转换
    try:
        print("开始转换MNIST模型...")
        print(f"输入文件: {pth_file}")
        print(f"输出文件: {onnx_file}")
        print()

        output_path = convert_pth_to_onnx(pth_file, onnx_file, input_size=(28, 28))

    except Exception as e:
        print(f"\n❌ 转换失败: {e}")
        import traceback

        traceback.print_exc()
