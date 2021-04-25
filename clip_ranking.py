import clip
import torch
from PIL import Image


# A helper function
def find_position(num, lst, lst_len):
    """
    Assert: lst is sorted from greatest to smallest; len(lst) == lst_len

    Returns the index of num in lst if it is insert to lst
    """
    try:
        idx = lst.index(num)
    except ValueError:
        for i in range(lst_len):
            if lst[i] < num:
                return i
        return -1
    else:
        return idx


class CLIP:
    """
    A text-label ranking system using CLIP developed by Open.ai.
    For paper please check: https://arxiv.org/pdf/2103.00020.pdf
    For official codebase please check: https://github.com/openai/CLIP
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)

    def get_score(self, img, label):
        """
        Return the score of the image corresponding to label according to the
        pretrained CLIP model.
        """
        image = self.preprocess(img).unsqueeze(0).to(self.device)
        text = clip.tokenize([label]).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)

            logits_per_image, logits_per_text = self.model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()

        return logits_per_image.cpu().numpy()[0, 0]

    def sample_selection(self, label_file, num_sample):
        """
        Returns the list of indices of images with highest score of match between
        image and label according to CLIP model.
        """
        labels = open(label_file, 'r', encoding='utf-8').readlines()
        best_samples = [-1] * num_sample
        best_scores = [0] * num_sample

        for i in range(len(labels)):
            score = self.get_score(Image.open('./102flowers/image_{:05d}.jpg'.format(i)), labels[i])
            if i == 0:
                best_samples[0] = 0
                best_scores[0] = score
            elif -1 in best_samples or (score > best_scores[-1]):
                idx = find_position(score, best_scores, num_sample)
                if idx == num_sample - 1:
                    best_scores[-1] = score
                    best_samples[-1] = i
                else:
                    best_scores = best_scores[:idx] + [score] + best_scores[idx: num_sample - 1]
                    best_samples = best_samples[:idx] + [i] + best_samples[idx: num_sample - 1]

        return best_samples


if __name__ == '__main__':
    # A example of CLIP in real practice
    clip_model = CLIP()
    img = Image.open('./image_00103.png')

    clip_model.get_score(img, 'A flower')
    clip_model.get_score(img, 'A flower with green leaves')
    clip_model.get_score(img, 'A flower with green leaves in front of blue background')
    clip_model.get_score(img, 'A blue, white and purple flower with green leaves in front of blue background')
    clip_model.get_score(img, 'University of Toronto')
