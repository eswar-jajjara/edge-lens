"""Known faults in disposable synthetic ONNX classifiers; no user model edits."""
import copy
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import TensorProto as T, helper as h, numpy_helper
from app.services.structural import onnx_inventory, compare_onnx
from app.services.benchmark import _ort_session, _hash_file


def fixture():
    nodes = [h.make_node('Mul', ['image', 'scale'], ['scaled'], name='scale'),
             h.make_node('Relu', ['scaled'], ['activated'], name='activation'),
             h.make_node('GlobalAveragePool', ['activated'], ['pooled'], name='pool'),
             h.make_node('Flatten', ['pooled'], ['flat'], name='flatten', axis=1),
             h.make_node('Gemm', ['flat', 'weight', 'bias'], ['logits'], name='classifier')]
    weights = [numpy_helper.from_array(np.array(1., dtype=np.float32), 'scale'),
               numpy_helper.from_array(np.array([[1., -1.], [.5, -.5], [.2, -.2]], dtype=np.float32), 'weight'),
               numpy_helper.from_array(np.zeros(2, dtype=np.float32), 'bias')]
    graph = h.make_graph(nodes, 'fault_validation_only', [h.make_tensor_value_info('image', T.FLOAT, [1, 3, 8, 8])],
                         [h.make_tensor_value_info('logits', T.FLOAT, [1, 2])], weights)
    model = h.make_model(graph, opset_imports=[h.make_opsetid('', 18)])
    model.ir_version = 10
    return model


def capture(model, sample):
    import onnxruntime as ort
    model = onnx.shape_inference.infer_shapes(copy.deepcopy(model))
    infos = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output)}
    names = [n.output[0] for n in model.graph.node]
    del model.graph.output[:]
    for name in names: model.graph.output.append(infos[name])
    session = _ort_session(ort, model.SerializeToString(), 1, diagnostic=True)
    return dict(zip(names, session.run(names, {'image': sample})))


def run_fault_suite(directory):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    baseline = fixture(); reference_path = directory / 'reference.onnx'; onnx.save(baseline, reference_path)
    reference = onnx_inventory(reference_path)
    sample = np.random.default_rng(73).uniform(-1, 1, (1, 3, 8, 8)).astype(np.float32)
    expected = capture(baseline, sample)
    rows = []
    for fault in ('unchanged_control', 'changed_weights', 'replaced_operator', 'rewired_connection', 'wrong_shape', 'removed_operator', 'wrong_preprocessing'):
        model = copy.deepcopy(baseline); candidate_input = sample.copy()
        expected_change = {'changed_weights': 'parameters', 'replaced_operator': 'operator', 'rewired_connection': 'connections'}
        if fault == 'changed_weights':
            weight = numpy_helper.to_array(model.graph.initializer[1]).copy(); weight[0, 0] += .5
            model.graph.initializer[1].CopyFrom(numpy_helper.from_array(weight, 'weight'))
        elif fault == 'replaced_operator': model.graph.node[1].op_type = 'Sigmoid'
        elif fault == 'rewired_connection': model.graph.node[2].input[0] = 'scaled'
        elif fault == 'wrong_shape': model.graph.output[0].type.tensor_type.shape.dim[1].dim_value = 3
        elif fault == 'removed_operator': del model.graph.node[1]
        elif fault == 'wrong_preprocessing': candidate_input *= .5
        path = directory / (fault + '.onnx'); onnx.save(model, path)
        row = {'fault': fault, 'artifact_sha256': _hash_file(path), 'evidence_status': 'MEASURED',
               'expected': 'no differences' if fault == 'unchanged_control' else 'rejection' if fault in {'wrong_shape', 'removed_operator'} else 'output drift',
               'first_observed_divergence': None, 'structural': None, 'error': None}
        try:
            onnx.checker.check_model(model, full_check=True)
            row['structural'] = compare_onnx(reference, onnx_inventory(path))
            actual = capture(model, candidate_input)
            numerical = []
            for node in baseline.graph.node:
                name = node.output[0]; ref, out = expected[name], actual[name]
                difference = np.abs(out.astype(np.float64) - ref)
                drift = not np.allclose(out, ref, atol=1e-4, rtol=1e-3)
                numerical.append({'name': node.name, 'status': 'drift' if drift else 'pass', 'mae': float(difference.mean()), 'max_abs': float(difference.max())})
                if drift and row['first_observed_divergence'] is None: row['first_observed_divergence'] = node.name
            row['numerical'] = numerical
            row['passed'] = row['first_observed_divergence'] is None if fault == 'unchanged_control' else row['first_observed_divergence'] is not None
            if fault in expected_change:
                row['passed'] &= any(expected_change[fault] in r['changes'] for r in row['structural']['rows'])
            if fault in {'wrong_shape', 'removed_operator'}: row['passed'] = False
        except Exception as exc:
            row['error'] = f'{type(exc).__name__}: {str(exc)[:500]}'
            row['passed'] = fault in {'wrong_shape', 'removed_operator'}
        rows.append(row)
    report = {'evidence_status': 'MEASURED', 'reference_sha256': _hash_file(reference_path), 'seed': 73,
              'scope': 'Synthetic disposable five-operation classifier; not trained-model quality evidence.',
              'passed': all(r['passed'] for r in rows), 'cases': rows,
              'limitation': 'First observed divergence is not causal proof. Bad preprocessing cannot always be inferred from model outputs without a declared input contract.'}
    (directory / 'fault-validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    from html import escape
    html = '<!doctype html><meta charset="utf-8"><title>EdgeLens fault validation</title><style>body{font:16px system-ui;max-width:1100px;margin:3rem auto}td,th{padding:12px;text-align:left;border-bottom:1px solid #ddd}pre{white-space:pre-wrap}</style><h1>Controlled fault validation · MEASURED</h1><p>' + escape(report['scope']) + '</p><table><tr><th>Fault</th><th>Expected</th><th>First observed difference</th><th>Result</th></tr>'
    for row in rows: html += '<tr>' + ''.join('<td>' + escape(str(x)) + '</td>' for x in (row['fault'], row['expected'], row['first_observed_divergence'] or row['error'] or 'None', 'PASS' if row['passed'] else 'FAIL')) + '</tr>'
    html += '</table><p>' + escape(report['limitation']) + '</p><details><summary>Full evidence</summary><pre>' + escape(json.dumps(report, indent=2)) + '</pre></details>'
    (directory / 'fault-validation.html').write_text(html, encoding='utf-8')
    return report
