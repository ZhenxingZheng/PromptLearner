import torch

def get_mask_sequence(texts, text_sequence):
    data = texts[:, 1:]
    text_sequence = text_sequence[:, 1:, :]
    mask = ((data !=0) & (data !=49407)).type(text_sequence.dtype).unsqueeze(2).cuda()
    text_embedding_sequence = text_sequence * mask

    return mask.squeeze(-1), text_embedding_sequence

if __name__ == '__main__':
    data = torch.ones(4, 11).long() + 1
    text_embeddings = torch.randn(4, 11, 512)
    data[:, 0] = 49406
    data[0, 5] = 49407
    data[0, 6:] = 0
    data[1, 7] = 49407
    data[1, 8:] = 0
    data[2, 6] = 49407
    data[2, 7:] = 0
    data[3, 8] = 49407
    data[3, 9:] = 0
    output, img_embedding_sequence = get_mask_sequence(data, text_embeddings)
    print(output)
    print(img_embedding_sequence.size())