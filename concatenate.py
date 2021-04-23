from bpe import *
from dvae_sample import *

if __name__ == '__main__':
    # # BPE
    # file = ['./caption.txt']
    # word_dict = get_word(file)
    #
    # print('==========')
    # print('Tokens Before BPE')
    # tokens_freq, tokens_dict = word_to_token(word_dict)
    # print('All tokens: {}'.format(tokens_freq.keys()))
    # print('Number of tokens: {}'.format(len(tokens_freq.keys())))
    # print('==========')
    #
    # token_limit = 256
    # while len(tokens_freq.keys()) < token_limit:
    #     pairs = get_stats(word_dict)
    #     if not pairs:
    #         break
    #     best = max(pairs, key=pairs.get)
    #     word_dict = merge_word(best, word_dict)
    #     # print('Iter: {}'.format(i))
    #     # print('Best pair: {}'.format(best))
    #     tokens_freq, tokens_dict = word_to_token(word_dict)
    #     print('All tokens: {}'.format(tokens_freq.keys()))
    #     print('Number of tokens: {}'.format(len(tokens_freq.keys())))
    #     print('==========')
    #
    # sorted_tokens_tuple = sorted(tokens_freq.items(), key=lambda item: (token_len(item[0]), item[1]), reverse=True)
    # sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
    #
    # print(sorted_tokens)
    #
    # word_given = 'blueandredflowerwithgreenpetal</w>'
    # print('Tokenizing word: {}...'.format(word_given))
    # if word_given in tokens_dict:
    #     print('Tokenization of the known word:')
    #     print(tokens_dict[word_given])
    #     print('Tokenization treating the known word as unknown:')
    #     print(tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>'))
    # else:
    #     print('Tokenizating of the unknown word:')
    #     result_tokens = tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='</u>')
    #     print(result_tokens)
    #     print(token_to_vector(sorted_tokens, result_tokens))

    # dVae
    dev = torch.device('cpu')
    enc = load_model("https://cdn.openai.com/dall-e/encoder.pkl", dev)

    x = preprocess(open_image('../UT/Course/Year 4/CSC412/Project/image/image_00001.jpg'))
    original_image = T.ToPILImage(mode='RGB')(x[0])
    # original_image.show()
    z_logits = enc(x)
    z = ((torch.argmax(z_logits, axis=1)).numpy())[0]


