import torch
import torch.nn as nn
from transformers import DistilBertModel
import torchvision



class Basic(nn.Module):
    def __init__(
            self, text_dim=1268+4, hidden_dim=200, img_dim=1000, rep_dim=500, output_dim=4):
        super(Basic, self).__init__()

        self.hidden_layer = nn.Linear(text_dim, hidden_dim)
        self.softmax = nn.Softmax(dim=-1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.bert = DistilBertModel.from_pretrained(
            "distilbert-base-uncased", return_dict=True)

        self.image_model = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)

        self.main = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

        self.image_main = nn.Sequential(
            nn.Linear(img_dim, rep_dim),
        )


    def forward(self, text_feat, image_embedding, action):
        image_embedding = self.image_main(image_embedding)
        image_embedding = image_embedding

        text_embedding = self.text_embed(text_feat)
        text_embedding = text_embedding.unsqueeze(0).expand(image_embedding.shape[0],-1)
        con_vec = torch.cat((text_embedding, image_embedding, action), 1)
        return self.main(con_vec)

    def text_embed(self, text):
        outputs = self.bert(**text)
        cls_token = outputs.last_hidden_state[:, -1, :]
        return cls_token.squeeze(0)

