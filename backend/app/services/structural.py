"""Conservative graph evidence. Unknown correspondence is never a pass."""
from collections import Counter
import hashlib
import re
import ast

ATEN_OPERATIONS = {'conv2d': {'Conv'}, 'convolution': {'Conv'}, 'relu': {'Relu'}, 'linear': {'Gemm'},
                   'add': {'Add'}, 'mul': {'Mul'}, 'sigmoid': {'Sigmoid'}, 'mean': {'ReduceMean'},
                   'adaptive_avg_pool2d': {'GlobalAveragePool', 'AveragePool'}, 'view': {'Reshape'}, 'reshape': {'Reshape'}, 'flatten': {'Flatten', 'Reshape'}}


def onnx_inventory(path):
    import onnx
    graph = onnx.shape_inference.infer_shapes(onnx.load(str(path), load_external_data=False)).graph
    shapes = {v.name: [d.dim_value if d.HasField('dim_value') else d.dim_param or None
                       for d in v.type.tensor_type.shape.dim]
              for v in list(graph.input) + list(graph.value_info) + list(graph.output)}
    producers = {t: n.name or f'node_{i}' for i, n in enumerate(graph.node) for t in n.output}
    weights = {v.name: hashlib.sha256(v.SerializeToString()).hexdigest() for v in graph.initializer}
    rows = []
    for i, node in enumerate(graph.node):
        props = {p.key: p.value for p in node.metadata_props}
        match = re.match(r'%([^ :]+)\s*:', props.get('pkg.torch.onnx.fx_node', ''))
        try: scopes = ast.literal_eval(props.get('pkg.torch.onnx.name_scopes', '[]'))[:-1]
        except (ValueError, SyntaxError, TypeError): scopes = []
        rows.append({'name': node.name or f'node_{i}', 'operation': node.op_type, 'domain': node.domain,
                     'origin': match.group(1) if match else None, 'module_scopes': scopes, 'inputs': list(node.input),
                     'outputs': list(node.output), 'parents': [producers.get(t, 'input:' + t) for t in node.input if t and t not in weights],
                     'output_shapes': [shapes.get(t) for t in node.output],
                     'attributes_sha256': hashlib.sha256(b''.join(a.SerializeToString() for a in node.attribute)).hexdigest(),
                     'weights': {t: weights[t] for t in node.input if t in weights}})
    return {'format': 'onnx', 'nodes': rows, 'inputs': {v.name: shapes.get(v.name) for v in graph.input},
            'outputs': {v.name: shapes.get(v.name) for v in graph.output}, 'evidence_status': 'MEASURED'}


def tflite_inventory(path):
    from ai_edge_litert import schema_py_generated as s
    model = s.Model.GetRootAsModel(path.read_bytes(), 0)
    names = {v: k for k, v in vars(s.BuiltinOperator).items() if isinstance(v, int)}
    rows, tensors = [], []
    tensor_types = {v: k for k, v in vars(s.TensorType).items() if isinstance(v, int)}
    for sub_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(sub_index)
        for tensor_index in range(graph.TensorsLength()):
            tensor = graph.Tensors(tensor_index); quant = tensor.Quantization()
            buffer = model.Buffers(tensor.Buffer())
            tensors.append({'subgraph': sub_index, 'index': tensor_index, 'name': tensor.Name().decode('utf-8', errors='replace') if tensor.Name() else '',
                            'shape': [int(x) for x in tensor.ShapeAsNumpy()], 'dtype': tensor_types.get(tensor.Type(), 'UNKNOWN'),
                            'constant_bytes': buffer.DataLength(), 'quantization_scales': [float(quant.Scale(j)) for j in range(quant.ScaleLength())] if quant else [],
                            'quantization_zero_points': [int(quant.ZeroPoint(j)) for j in range(quant.ZeroPointLength())] if quant else []})
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i); code = model.OperatorCodes(op.OpcodeIndex())
            inputs = [int(op.Inputs(j)) for j in range(op.InputsLength())]
            outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
            rows.append({'name': f'subgraph_{sub_index}/op_{i}', 'operation': names.get(code.BuiltinCode(), 'CUSTOM'),
                         'inputs': inputs, 'outputs': outputs,
                         'output_shapes': [[int(x) for x in graph.Tensors(t).ShapeAsNumpy()] for t in outputs],
                         'comparison_status': 'UNAVAILABLE', 'reason_code': 'cross_format_mapping_unavailable'})
    return {'format': 'tflite', 'nodes': rows, 'tensors': tensors,
            'tensor_dtype_counts': dict(Counter(t['dtype'] for t in tensors)), 'evidence_status': 'MEASURED',
            'limitation': 'Flatbuffer operator/tensor inventory; no PyTorch numerical identity is inferred.'}


def compare_onnx(reference, candidate):
    """Exact unique node identities; Q/DQ helpers are transparent only for edges."""
    ref, out = reference['nodes'], candidate['nodes']
    counts = Counter(n['name'] for n in ref); other_counts = Counter(n['name'] for n in out)
    by_name = {n['name']: n for n in out}; helpers = {'QuantizeLinear', 'DequantizeLinear'}
    tensor_parent = {t: n for n in out for t in n['outputs']}
    def parent(tensor, visited=None):
        visited = set() if visited is None else visited
        if tensor in visited: return 'cycle:' + tensor
        visited.add(tensor)
        node = tensor_parent.get(tensor)
        if node and node['operation'] in helpers and node['inputs']:
            return parent(node['inputs'][0], visited)
        return node['name'] if node else 'input:' + tensor
    rows = []
    for node in ref:
        target = by_name.get(node['name'])
        changes = []
        if counts[node['name']] != 1 or other_counts[node['name']] > 1:
            status, reason = 'UNAVAILABLE', 'ambiguous_node_identity'
        elif target is None:
            status, reason = 'UNAVAILABLE', 'removed_or_fused_operation'
        else:
            candidate_parents = [parent(t) for t in target['inputs'] if t and t not in target['weights']]
            if target['operation'] != node['operation'] or target['domain'] != node['domain']: changes.append('operator')
            if target['output_shapes'] != node['output_shapes']: changes.append('shape')
            if candidate_parents != node['parents']: changes.append('connections')
            if target['attributes_sha256'] != node['attributes_sha256']: changes.append('attributes')
            if target['weights'] != node['weights']: changes.append('parameters')
            status, reason = ('CHANGED', 'mapped_graph_difference') if changes else ('MATCHED', 'unique_preserved_node')
        rows.append({'name': node['name'], 'operation': node['operation'], 'status': status,
                     'reason_code': reason, 'changes': changes, 'expected_shape': node['output_shapes'],
                     'actual_shape': target['output_shapes'] if target else None,
                     'expected_parents': node['parents'], 'actual_parents': candidate_parents if target and status != 'UNAVAILABLE' else None})
    extras = [n for n in out if n['name'] not in counts]
    return {'evidence_status': 'MEASURED', 'rows': rows, 'matched': sum(r['status'] == 'MATCHED' for r in rows),
            'changed': sum(r['status'] == 'CHANGED' for r in rows), 'unavailable': sum(r['status'] == 'UNAVAILABLE' for r in rows),
            'added_operations': [{'name': n['name'], 'operation': n['operation'], 'classification': 'quantization_helper' if n['operation'] in helpers else 'unmatched_operation'} for n in extras],
            'limitation': 'Structural differences are observations, not proof of invalid conversion. Fusions/decompositions need an explicit mapping rule; parameter changes may be expected quantization.'}


def fx_inventory(ep):
    rows = []
    for node in ep.module().graph.nodes:
        value = node.meta.get('val')
        rows.append({'name': node.name, 'operation': str(node.target), 'kind': node.op,
                     'module_scopes': [v[0] for v in node.meta.get('nn_module_stack', {}).values()],
                     'parents': [n.name for n in node.all_input_nodes],
                     'output_shape': [str(d) if not isinstance(d, int) else d for d in value.shape] if hasattr(value, 'shape') else None})
    return {'format': 'pt2', 'nodes': rows, 'evidence_status': 'MEASURED'}


def fx_boundaries(reference, candidate):
    """Exact origins, plus two bounded batch-one scope/shape decomposition rules."""
    sources = [n for n in reference['nodes'] if n['kind'] == 'call_function']
    scopes = Counter(tuple(n.get('module_scopes', [])) for n in sources)
    result = {}
    for source in sources:
        matches = [n for n in candidate['nodes'] if n['origin'] == source['name']]
        method = 'exact_exporter_origin'
        operation = source['operation'].split('.')[-2] if '.' in source['operation'] else source['operation']
        if not matches and operation in {'adaptive_avg_pool2d', 'flatten'} and scopes[tuple(source.get('module_scopes', []))] == 1 and any(source.get('module_scopes', [])):
            supported = {'ReduceMean', 'GlobalAveragePool'} if operation == 'adaptive_avg_pool2d' else {'Flatten', 'Reshape'}
            shape = source['output_shape']
            shape_supported = shape and shape[0] == 1 and (operation == 'flatten' and len(shape) == 2 or operation == 'adaptive_avg_pool2d' and len(shape) == 4 and shape[-2:] == [1, 1])
            if shape_supported:
                matches = [n for n in candidate['nodes'] if n.get('module_scopes') == source.get('module_scopes') and n['operation'] in supported
                           and n['domain'] in {'', 'ai.onnx'} and n['output_shapes'] == [shape]]
                method = 'unique_module_scope_and_supported_output_shape_rule'
        target = matches[0] if len(matches) == 1 else None
        allowed = ATEN_OPERATIONS.get(operation, set())
        if method != 'exact_exporter_origin': allowed = {'ReduceMean', 'GlobalAveragePool'} if operation == 'adaptive_avg_pool2d' else {'Flatten', 'Reshape'}
        valid = target and target['domain'] in {'', 'ai.onnx'} and target['operation'] in allowed and len(target['outputs']) == 1
        result[source['name']] = {'target': target if valid else None, 'observed_target': target,
                                  'method': method, 'reason_code': 'mapped_boundary' if valid else 'no_exporter_origin' if not matches else 'decomposed_operation' if len(matches) > 1 else 'unsupported_operator_mapping'}
    return result


def compare_fx_onnx(reference, candidate):
    origins = {}
    for node in candidate['nodes']:
        if node['origin']: origins.setdefault(node['origin'], []).append(node)
    by_name = {n['name']: n for n in candidate['nodes']}
    boundaries = fx_boundaries(reference, candidate)
    target_to_source = {item['target']['name']: name for name, item in boundaries.items() if item['target']}
    placeholder_names = [n['name'] for n in reference['nodes'] if n['kind'] == 'placeholder']
    input_aliases = dict(zip(candidate.get('inputs', {}), placeholder_names)) if len(candidate.get('inputs', {})) == len(placeholder_names) == 1 else {}
    rows = []
    for source in reference['nodes']:
        if source['kind'] != 'call_function': continue
        boundary = boundaries[source['name']]
        matches = [boundary['target']] if boundary['target'] else origins.get(source['name'], [])
        row = {'name': source['name'], 'operation': source['operation'], 'expected_shape': source['output_shape'],
               'expected_parents': source['parents'], 'mapped_operations': [n['name'] for n in matches],
               'status': 'UNAVAILABLE', 'reason_code': 'no_exporter_origin', 'changes': []}
        if len(matches) > 1:
            row['reason_code'] = 'decomposed_operation'
        elif len(matches) == 1:
            target = matches[0]
            aten_operation = source['operation'].split('.')[-2] if '.' in source['operation'] else source['operation']
            supported_operations = dict(ATEN_OPERATIONS)
            if boundary['method'] != 'exact_exporter_origin': supported_operations['adaptive_avg_pool2d'] = {'ReduceMean', 'GlobalAveragePool'}
            operator_known = aten_operation in supported_operations
            if operator_known and target['operation'] not in supported_operations[aten_operation]: row['changes'].append('operator')
            row['actual_shape'] = target['output_shapes'][0] if len(target['output_shapes']) == 1 else None
            # Shape/axes construction is not a data connection for these declared rules.
            target_parents = target['parents'][:1] if aten_operation in {'flatten', 'view', 'reshape', 'adaptive_avg_pool2d', 'mean'} else target['parents']
            parent_origins = [target_to_source.get(p, by_name[p]['origin']) if p in by_name else input_aliases.get(p.removeprefix('input:')) for p in target_parents]
            row['actual_parents'] = parent_origins
            expected_nodes = [n for n in reference['nodes'] if n['name'] in source['parents']]
            expected_data = [n['name'] for n in expected_nodes if n['kind'] != 'get_attr']
            shape_known = row['actual_shape'] is not None and row['expected_shape'] is not None
            valid_source_names = {n['name'] for n in reference['nodes']}
            edges_known = all(p in valid_source_names for p in parent_origins)
            if shape_known and row['actual_shape'] != row['expected_shape']: row['changes'].append('shape')
            if edges_known and parent_origins != expected_data: row['changes'].append('connections')
            row.update(status='CHANGED' if row['changes'] else 'MATCHED' if shape_known and edges_known and operator_known else 'UNAVAILABLE',
                       mapping_method=boundary['method'], reason_code='exporter_origin_boundary' if shape_known and edges_known and operator_known else 'unsupported_operator_mapping' if not operator_known else 'shape_or_parent_mapping_unavailable')
        rows.append(row)
    return {'reference': 'pytorch_fx', 'candidate': 'onnx', 'evidence_status': 'MEASURED', 'rows': rows,
            'matched': sum(r['status'] == 'MATCHED' for r in rows), 'changed': sum(r['status'] == 'CHANGED' for r in rows),
            'unavailable': sum(r['status'] == 'UNAVAILABLE' for r in rows),
            'limitation': 'Exporter-origin correspondence is bounded evidence. CHANGED connections may be legal decomposition or optimization; numerical verification is separate.'}
