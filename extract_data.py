import os
import cv2
import tqdm
import numpy as np


# def load_data():
#     X, Y = {}, {}
#     c = 0
#     for s in ["train", "valid"]:
#         if  os.path.exists("data/" + s + ".npz"):
#             files = np.load(open("data/" + s + ".npz", "r"))
#             X[s], Y[s] = [files["X0"], files["X1"]], files["Y"]
#         else:
#             X[s], Y[s] = [[], []], []
#             with open("data/" + s + ".txt", "r") as file:
#                 for line in tqdm.tqdm(file.readlines(), desc=s):
#                     elements = line.strip().split()
#                     prefix, a, b = map(int, elements[:3])
#                     for i, x in enumerate([a, b]):
#                         image = cv2.imread(
#                             "data/train_val/train_val_imgs/{:06d}-{:02d}.JPG".format(prefix, x))
#                         width, height = image.shape[:2]
#                         if width < height:
#                             width = int(224. * width / height)
#                             height = 224
#                         else:
#                             height = int(224. * height / width)
#                             width = 224
#                         image = cv2.resize(
#                             image, (height, width)).astype(np.float32)
#                         image = np.pad(image, ((
#                             0, 224 - width), (0, 224 - height), (0, 0)), mode="constant", constant_values=0)
#                         X[s][i].append(image / 255.)
#                     if s in ["train", "valid"]:
#                         score = float(elements[3])
#                         if score < 0.5:
#                             Y[s].append([1, 0])
#                         else:
#                             Y[s].append([0, 1])
#                         if s == "valid" and score >= 0.3 and score <= 0.7:
#                             X[s][0].pop()
#                             X[s][1].pop()
#                             Y[s].pop()
#             X[s][0], X[s][1], Y[s] = map(
#                 np.array, [X[s][0], X[s][1], Y[s]])
#             np.savez(open("data/" + s + ".npz", "w"),
#                      X0=X[s][0], X1=X[s][1], Y=Y[s])
#     return X, Y


def load_data(path):
    files = np.load(open(path, 'r'))
    return (files['X0'], files['X1'], files['Y'])


def resize_pad_normalize(image):
    width, height = image.shape[:2]
    width, height = (int(224. * width / height), 224) if width < height else (224, int(224. * height / width))
    image = cv2.resize(
        image, (height, width)).astype(np.float32)
    image = np.pad(
        image, ((0, 224 - width), (0, 224 - height), (0, 0)), mode="constant", constant_values=0)
    return image / 255.


def extract_data(path):
    X, Y = [[], []], []
    c = 0 
    with open(path, 'r') as file:
        for line in tqdm.tqdm(file.readlines(), desc=('loading %s' % path)):
            c += 1
            elements = line.strip().split()
            prefix, a, b = map(int, elements[:3])
            image_a = cv2.imread(
                'dataset/train_val/train_val_imgs/{:06d}-{:02d}.JPG'.format(prefix, a))
            image_b = cv2.imread(
                'dataset/train_val/train_val_imgs/{:06d}-{:02d}.JPG'.format(prefix, b))
            X[0].append(resize_pad_normalize(image_a))
            X[1].append(resize_pad_normalize(image_b))
            score = float(elements[3])
            score_vector = -1 if score < 0.5 else 1
            Y.append(score_vector)
            if 'valid' in path and score >= 0.3 and score <= 0.7:
                X[0].pop()
                X[1].pop()
                Y.pop()
            if c == 5000:
                break
    X[0], X[1], Y = map(np.array, [X[0], X[1], Y])
    np.savez(open(path.split(".")[0] + ".npz", "w"), X0=X[0], X1=X[1], Y=Y)


def main():
    train_path = 'dataset/train'
    valid_path = 'dataset/val'
    train_data = load_data('%s.npz' % train_path) if os.path.exists(
        '%s.npz' % train_path) else extract_data('%s.txt' % train_path)
    valid_data = load_data('%s.npz' % valid_path) if os.path.exists(
        '%s.npz' % valid_path) else extract_data('%s.txt' % valid_path)
    return (train_data, valid_data)


if __name__ == '__main__':
    train_data, valid_data = main()
    # print X, Y 
