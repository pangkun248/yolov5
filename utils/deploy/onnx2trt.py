import tensorrt as trt
from calibrator import YOLOEntropyCalibrator
import onnxruntime
import os

TRT_LOGGER = trt.Logger()
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def onnx2trt(onnx_path, trt_path, int8=False, dynamic=None, img_path=''):
    os.system('rm calib_yolo.bin')
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
            EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()

        # 各种layer算法通常需要临时工作空间。这个参数限制了网络中所有的层可以使用的最大的workspace空间大小。
        # 如果分配的空间不足，TensorRT可能无法找到给定层的实现
        config.max_workspace_size = 1 << 30
        builder.max_batch_size = 1  # 指定TensorRT将要优化的batch大小.在运行时,只能选择比这个值小的batch 如果最终bs维度不为1 则此处需要更改
        if dynamic:
            profile = builder.create_optimization_profile()
            # 默认为静态输入.动态输入可参考 https://blog.csdn.net/qq_36276587/article/details/113175314
            # set_shape第一个参数必须与onnx.get_inputs()[0].name一致.
            profile.set_shape(
                'images',  # input tensor name
                *dynamic)
            config.add_optimization_profile(profile)

        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
            # 该方法是为了进行INT8量化时需要用到的校准器,如果出现WARNING: Missing dynamic range for tensor相关的警告时,
            # 将calib_yolo.bin删除再运行即可,否则int8权重会出现fp32的值导致模型混乱,严重拖慢trt模型的速度 这里640也可以是其他值
            config.int8_calibrator = YOLOEntropyCalibrator(img_path, (640, 640), 'calib_yolo.bin', batch_size=1)
            if dynamic:
                # TensorRT优化以下配置
                config.set_calibration_profile(profile)
        else:
            config.set_flag(trt.BuilderFlag.FP16)
        print('正在解析ONNX文件 {}...'.format(onnx_path))
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print('错误: 解析ONNX文件失败.')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                exit()
        print('解析完成,正在构建TensorRT Engine {},这大概需要一会时间...'.format(trt_path))
        engine = builder.build_engine(network, config)
        with open(trt_path, "wb") as t:
            t.write(engine.serialize())
        print("TensorRT Engine 构建完成")


if __name__ == '__main__':
    """
    执行该脚本前,你需要确定你的onnx模型的输入shape是否是动态的,v5提供的export默认为静态.
    """
    is_int8 = False
    # 默认动态范围shape  需根据实际应用场景进行设置,否则当实际输入shape越界时context.get_binding_shape(1)返回(0),即无法根据输入推出输出shape
    dynamic_parm = [
        [1, 3, 160, 160],  # 各维度最小输入shape
        [1, 3, 640, 640],  # 各维度最常输入shape
        [2, 3, 640, 640],  # 各维度最大输入shape
    ]
    onnx_path = '/home/cmv/PycharmProjects/yolov5/yolov5s.onnx'
    img_dir = '/home/cmv/PycharmProjects/datasets/test'
    session = onnxruntime.InferenceSession(onnx_path)
    input_shape = session.get_inputs()[0].shape  # 根据onnx中输入shape是否为字符串来判断是否动态
    if list(map(lambda x: isinstance(x, int), input_shape)) == [True, True, True, True]:
        dynamic_parm = None
        save_path = onnx_path.replace('.onnx', '_stable_int8.trt' if is_int8 else '_stable_fp16.trt')
    else:
        dynamic_name = ''
        for i, shape_ in enumerate(input_shape):
            dim = '_-1'
            if isinstance(shape_, int):  # 可能bs 或 height 或 width实际输入为固定的,所以这里这里动态范围也进行固定
                dim = '_' + str(shape_)
                dynamic_parm[0][i] = shape_
                dynamic_parm[1][i] = shape_
                dynamic_parm[2][i] = shape_
            dynamic_name += dim
        save_path = onnx_path.replace('.onnx', dynamic_name + ('_int8.trt' if is_int8 else '_fp16.trt'))
    onnx2trt(onnx_path, save_path, int8=is_int8, dynamic=dynamic_parm, img_path=img_dir)
