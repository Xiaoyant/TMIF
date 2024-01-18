from torch import nn

from models import resnet
from models.ImgTransformer import ImgBlock, CrossAttention
from models.TextTransformer import TextBlock


class CrossTransIngration(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_size):
        super(CrossTransIngration, self).__init__()
        self.dim = embedding_size
        # resnet
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_classes)
        # img layer
        self.img_conv = nn.Conv2d(2048, self.dim, kernel_size=1)
        # text Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, self.dim)
        # attn layer
        self.imgtransformer_encoder = ImgBlock(dim=self.dim)
        self.texttransformer_encoder = TextBlock(dim=self.dim)
        self.cross_attn = CrossAttention()
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 45)

    def forward(self, input_i, input_t):
        batch = input_t.size(0)
        resnet_out, rpn_feature, feature = self.pretrained_model(input_i)
        res_logits = resnet_out

        #  img layer   56 57 58 59   3000词库   62 （5000）
        img_feature = self.img_conv(rpn_feature)  # img_feature:[8,128,14,14]
        img_atten = self.imgtransformer_encoder(img_feature)  # img_atten:[8,128,14,14]
        img_avg = self.avgpool2(img_atten).view(batch, -1)  # [8,128]
        img_logits = self.fc(img_avg)

        # text Embedding layer
        text_embedding = self.embedding_layer(input_t).permute(0, 2, 1)  # [8,128,100]
        text_feature = text_embedding.reshape(batch, self.dim, 10, 10)
        text_atten = self.texttransformer_encoder(text_feature)  # text_atten:[8,128,10,10]
        text_avg = self.avgpool2(text_atten).view(batch, -1)  # [8,128]
        text_logits = self.fc(text_avg)

        # cross attention
        img_cross, text_cross = self.cross_attn(img_atten, text_atten)  # [8, 196, 128]
        mul_feature = img_cross.permute((0, 2, 1))  # [8, 128, 196]
        mul_feature = self.avgpool1(mul_feature).reshape(batch, -1)  # [8, 128]
        mul_logits = self.fc(mul_feature)

        return img_logits, res_logits, text_logits, mul_logits
