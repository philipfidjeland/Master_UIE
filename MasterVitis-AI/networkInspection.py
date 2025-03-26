from tensorflow_model_optimization.quantization.keras import vitis_inspect
import tensorflow as tf

model = tf.keras.models.load_model('modelSimple512_1epoch.h5')

inspector = vitis_inspect.VitisInspector(target="/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json")
inspector.inspect_model(model,
                       input_shape=[None ,512,  512, 3],
                       plot=True,
                       plot_file="float_model.svg",
                       dump_results=True,
                       dump_results_file="inspect_results.txt",
                       verbose=2)
