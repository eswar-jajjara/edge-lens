"""Linux-only FP32 and calibrated static INT8 conversion, in one environment."""
import copy
import platform
import time


def convert_profiles(ep, sample, calibration_arrays, output_dir):
    if platform.system() != 'Linux':
        raise ValueError('PT2 → TFLite conversion requires the separate pinned Linux/WSL worker. Windows supports evaluated TFLite imports.')
    import torch
    import litert_torch
    from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e
    from litert_torch.quantize.pt2e_quantizer import PT2EQuantizer, get_symmetric_quantization_config
    from litert_torch.quantize.quant_config import QuantConfig
    if not calibration_arrays:
        raise ValueError('Static INT8 TFLite conversion requires separate calibration images')
    profiles = []
    for name in ('standard', 'dashboard'):
        started = time.perf_counter()
        target = output_dir / (name + '.tflite')
        module = copy.deepcopy(ep).module()
        if name == 'standard':
            edge = litert_torch.convert(module, (sample,))
            precision = 'fp32'
        else:
            quantizer = PT2EQuantizer().set_global(get_symmetric_quantization_config(is_per_channel=True, is_dynamic=False))
            prepared = prepare_pt2e(torch.export.export(module, (sample,)).module(), quantizer)
            with torch.no_grad():
                for value in calibration_arrays: prepared(torch.from_numpy(value))
            quantized = convert_pt2e(prepared, fold_quantize=False)
            edge = litert_torch.convert(quantized, (sample,), quant_config=QuantConfig(pt2e_quantizer=quantizer))
            precision = 'static_int8'
        edge.export(str(target))
        profiles.append({'profile': name, 'path': target, 'precision': precision,
                         'conversion_seconds': time.perf_counter() - started,
                         'label': 'LiteRT FP32 control' if name == 'standard' else 'LiteRT calibrated static INT8',
                         'configuration': {'converter': 'litert-torch', 'quantization': precision,
                                           'calibration_samples': len(calibration_arrays) if name == 'dashboard' else 0}})
    return profiles
