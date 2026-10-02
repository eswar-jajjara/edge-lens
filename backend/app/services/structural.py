"""Conservative graph evidence. Unknown correspondence is never a pass."""
from collections import Counter
import hashlib
import re


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
        rows.append({'name': node.name or f'node_{i}', 'operation': node.op_type, 'domain': node.domain,
                     'origin': match.group(1) if match else None, 'inputs': list(node.input),
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
    rows = []
    for sub_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(sub_index)
        for i in range(graph.OperatorsLength()):
            op = graph.Operators(i); code = model.OperatorCodes(op.OpcodeIndex())
            inputs = [int(op.Inputs(j)) for j in range(op.InputsLength())]
            outputs = [int(op.Outputs(j)) for j in range(op.OutputsLength())]
            rows.append({'name': f'subgraph_{sub_index}/op_{i}', 'operation': names.get(code.BuiltinCode(), 'CUSTOM'),
                         'inputs': inputs, 'outputs': outputs,
                         'output_shapes': [[int(x) for x in graph.Tensors(t).ShapeAsNumpy()] for t in outputs],
                         'comparison_status': 'UNAVAILABLE', 'reason_code': 'cross_format_mapping_unavailable'})
    return {'format': 'tflite', 'nodes': rows, 'evidence_status': 'MEASURED',
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
                     'parents': [n.name for n in node.all_input_nodes],
                     'output_shape': [str(d) if not isinstance(d, int) else d for d in value.shape] if hasattr(value, 'shape') else None})
    return {'format': 'pt2', 'nodes': rows, 'evidence_status': 'MEASURED'}


def compare_fx_onnx(reference, candidate):
    origins = {}
    for node in candidate['nodes']:
        if node['origin']: origins.setdefault(node['origin'], []).append(node)
    by_name = {n['name']: n for n in candidate['nodes']}
    rows = []
    for source in reference['nodes']:
        if source['kind'] != 'call_function': continue
        matches = origins.get(source['name'], [])
        row = {'name': source['name'], 'operation': source['operation'], 'expected_shape': source['output_shape'],
               'expected_parents': source['parents'], 'mapped_operations': [n['name'] for n in matches],
               'status': 'UNAVAILABLE', 'reason_code': 'no_exporter_origin', 'changes': []}
        if len(matches) > 1:
            row['reason_code'] = 'decomposed_operation'
        elif len(matches) == 1:
            target = matches[0]
            row['actual_shape'] = target['output_shapes'][0] if len(target['output_shapes']) == 1 else None
            parent_origins = [by_name[p]['origin'] if p in by_name else p.removeprefix('input:') for p in target['parents']]
            row['actual_parents'] = parent_origins
            expected_nodes = [n for n in reference['nodes'] if n['name'] in source['parents']]
            expected_data = [n['name'] for n in expected_nodes if n['kind'] != 'get_attr']
            shape_known = row['actual_shape'] is not None and row['expected_shape'] is not None
            edges_known = all(p is not None for p in parent_origins)
            if shape_known and row['actual_shape'] != row['expected_shape']: row['changes'].append('shape')
            if edges_known and parent_origins != expected_data: row['changes'].append('connections')
            row.update(status='CHANGED' if row['changes'] else 'MATCHED' if shape_known and edges_known else 'UNAVAILABLE',
                       reason_code='exporter_origin_boundary' if shape_known and edges_known else 'shape_or_parent_mapping_unavailable')
        rows.append(row)
    return {'reference': 'pytorch_fx', 'candidate': 'onnx', 'evidence_status': 'MEASURED', 'rows': rows,
            'matched': sum(r['status'] == 'MATCHED' for r in rows), 'changed': sum(r['status'] == 'CHANGED' for r in rows),
            'unavailable': sum(r['status'] == 'UNAVAILABLE' for r in rows),
            'limitation': 'Exporter-origin correspondence is bounded evidence. CHANGED connections may be legal decomposition or optimization; numerical verification is separate.'}
