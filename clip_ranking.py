import torch
import clip
import numpy as np


class CLIP:
    """
    A text-label ranking system using CLIP developed by Open.ai.
    For paper please check: https://arxiv.org/pdf/2103.00020.pdf
    For official codebase please check: https://github.com/openai/CLIP

    To initialize:
    labels - The list containing all potential labels for CLIP to rank.
    """
    def __init__(self, labels):
        self.labels = labels
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_token = clip.tokenize(labels).to(self.device)
        self.text_features = self.model.encode_text(self.text_token)

    def get_label_for_image(self, img):
        """
        Returns the text label for img with the highest ranking.
        """
        image = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)

            logits_per_image, logits_per_text = self.model(image, self.text_token)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        max_index = np.argmax(probs)
        return self.labels[max_index]
