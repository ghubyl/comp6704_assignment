from pathlib import Path
import torch
from PIL import Image
from torchvision import transforms

project = Path(__file__).resolve().parents[1]
data_root = project / 'data' / 'minc-2500'
output_file = data_root.resolve().parent.parent / 'features' / 'minc_dinov2_cls_patch.pt'
batch_size = 32

categories = (data_root / 'categories.txt').read_text().splitlines()
class_to_index = {name: index for index, name in enumerate(categories)}
image_files = sorted((data_root / 'images').glob('*/*.jpg'))

model = torch.hub.load(str(project / 'vendor/dinov2'), 'dinov2_vits14',
                       source='local', pretrained=False)
weights = torch.load(project / 'checkpoints/dinov2_vits14_pretrain.pth',
                     map_location='cpu', weights_only=True)
model.load_state_dict(weights)
model = model.eval().cuda()

preprocess = transforms.Compose([
    transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

features, labels, image_paths = [], [], []
with torch.inference_mode():
    for start in range(0, len(image_files), batch_size):
        images = []
        for path in image_files[start:start + batch_size]:
            with Image.open(path) as image:
                images.append(preprocess(image.convert('RGB')))
            labels.append(class_to_index[path.parent.name])
            image_paths.append(path.relative_to(data_root).as_posix())

        outputs = model.forward_features(torch.stack(images).cuda())

        batch_features = torch.cat([
            outputs['x_norm_clstoken'],
            outputs['x_norm_patchtokens'].mean(dim=1),
        ], dim=1)

        batch_features = torch.nn.functional.normalize(batch_features, dim=1)
        features.append(batch_features.cpu())

output_file.parent.mkdir(parents=True, exist_ok=True)
torch.save({
    'features': torch.cat(features),
    'labels': torch.tensor(labels),
    'image_paths': image_paths,
    'class_to_index': class_to_index,
}, output_file)
print(f'Saved features to {output_file}')
