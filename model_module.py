
import torch.nn as nn
from transformers import ViTModel

model_name_or_path = 'google/vit-base-patch32-224-in21k'
    
class VITClassifier(nn.Module):
    def __init__(self, model_name , num_classes, freeze_layers = False):
        super(VITClassifier, self).__init__()
        self.model_name = model_name
        self.vit = ViTModel.from_pretrained(model_name)

        if freeze_layers:
            for params in self.vit.parameters():
                params.requires_grad = False
                #freezing the vit parameters as training vit is too time consuming

            # Unfreeze the last 2 layers for fine-tuning
            for layer in self.vit.encoder.layer[-2:]:
                for param in layer.parameters():
                    param.requires_grad = True
        
        self.classifier  = nn.Sequential(
            nn.Linear(self.vit.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
        
    def forward(self, x):
        output_vit = self.vit(x)
        output_logits = self.classifier(output_vit.last_hidden_state[:, 0])
        return output_logits
    
