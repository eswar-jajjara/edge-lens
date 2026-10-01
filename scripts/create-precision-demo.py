"""Generate a trained synthetic classifier and disjoint splits for workflow checks.

These colour images are deliberately simple. Their accuracy is not evidence of
real-world model quality, converter superiority, or physical-device performance.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile


def samples(seed, count):
    import numpy as np
    rng = np.random.default_rng(seed)
    values, labels = [], []
    for index in range(count):
        label = index % 2
        image = rng.integers(0, 70, (8, 8, 3), dtype=np.uint8)
        image[:, :, 2 if label else 0] += 150
        values.append(image)
        labels.append(label)
    return values, labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing files are never replaced')
    args = parser.parse_args()
    import numpy as np
    from PIL import Image
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(71)
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    model = torch.nn.Sequential(torch.nn.Conv2d(3, 8, 3, padding=1), torch.nn.ReLU(),
                                torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(8, 2))
    images, labels = samples(11, 64)
    train_x = torch.from_numpy(np.stack(images).astype(np.float32).transpose(0, 3, 1, 2) / 255)
    train_y = torch.tensor(labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=.03)
    model.train()
    for _ in range(40):
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(train_x), train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    torch.export.save(torch.export.export(model, (torch.zeros(1, 3, 8, 8),)), directory / 'colour-classifier.pt2')
    spec = {'name': 'Synthetic colour classifier · workflow demo', 'format': 'pt2', 'input_shape': [1, 3, 8, 8],
            'layout': 'NCHW', 'class_count': 2, 'scale': 1 / 255, 'mean': [0, 0, 0], 'std': [1, 1, 1], 'resize': 'stretch'}
    (directory / 'spec.json').write_text(json.dumps(spec, indent=2), encoding='utf-8')
    counts = {'calibration': 100, 'validation': 100, 'test': 300}
    for (role, count), seed in zip(counts.items(), (17, 29, 43)):
        images, labels = samples(seed, count)
        with zipfile.ZipFile(directory / (role + '.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('labels.json', json.dumps({'red': 0, 'blue': 1}))
            for index, (pixels, label) in enumerate(zip(images, labels)):
                raw = io.BytesIO(); Image.fromarray(pixels).save(raw, format='PNG')
                archive.writestr(f'{"blue" if label else "red"}/{role}-{index:04d}.png', raw.getvalue())
    (directory / 'README.txt').write_text('SYNTHETIC WORKFLOW DEMO ONLY\nTraining seed: 11. Calibration/validation/test seeds: 17/29/43.\n'
                                        'These images verify actual conversion, inference and report plumbing.\n'
                                        'Do not present their accuracy as real-world classification evidence or guaranteed optimization gains.\n', encoding='utf-8')
    print(f'Synthetic workflow files created in {directory}')
    print('Upload the PT2 with shape 1,3,8,8 and two classes; select Compare FP32 vs static INT8.')
    print('Use the three ZIPs in their named roles. Use real labelled data for your project evaluation.')


if __name__ == '__main__':
    main()
