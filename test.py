import os
import yaml
import torch
from PIL import Image
from torchvision import transforms
from dataManagement.DatasetHelper import DatasetHelper
from models.CrossTransIngration import CrossTransIngration

transform_test = transforms.Compose([
    transforms.Resize((448, 448), Image.BILINEAR),
    transforms.ToTensor(),
])
mul_model = CrossTransIngration(num_classes=45, vocab_size=3000, embedding_size=128)
base_dir = "./runs"
file = "/weights"
bestmodal = "/015.ckpt"
checkpoint = torch.load(base_dir + file + bestmodal)
save_dir = base_dir + file
mul_model.load_state_dict(checkpoint['net_state_dict'])
mul_model.eval()
with open('./OFG.yaml', 'r') as file:
    class_dict = yaml.load(file, Loader=yaml.FullLoader)['ofg_classes']

train_muldata_path = './data/mul_datasets/mul_train.txt'
test_muldata_path = './data/mul_datasets/mul_test.txt'

train_images = []
train_texts = []
with open(train_muldata_path) as tr:
    for line in tr.readlines():
        line = line.replace('\n', '')
        line = line.split('|')
        train_images.append(line[1])
        train_texts.append(line[2])

test_images = []
test_texts = []
with open(test_muldata_path) as tr:
    for line in tr.readlines():
        line = line.replace('\n', '')
        line = line.split('|')
        test_images.append(line[1])
        test_texts.append(line[2])

data_helper = DatasetHelper(100)
train_t, test_t = data_helper.preprocess_texts(train_texts, test_texts, 100)
acc = 0
total = len(test_images)
with open(os.path.join(save_dir, 'test.txt'), 'w') as output:
    for i in range(len(test_images)):
        img = Image.open(test_images[i])
        img = img.convert('RGB')

        parts = test_images[i].split("/")
        last_part = parts[-1]
        imgname = last_part[:-8]
        imgname = imgname.split("_")[0]

        scaled_img = transform_test(img)
        torch_image = scaled_img.unsqueeze(0)
        torch_text = test_t[i].unsqueeze(0)
        with torch.no_grad():
            img_logirs, res_logits, text_logits, mul_logits = mul_model(torch_image, torch_text)
            _, predict = torch.max(mul_logits, 1)
            pred_id = predict.item()
            if class_dict[pred_id] == imgname[:len(class_dict[pred_id])]:
                acc = acc + 1
                print(class_dict[pred_id])
    output.write("acnumc:" + str(acc) + "\n")
    output.write("total:" + str(total) + "\n")
    output.write("accnum/total" + str(acc / total) + "\n")
print(acc)
print(total)
print(acc / total)
