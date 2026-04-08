import argparse
import json

import cv2
import torch
import numpy as np
from skimage import measure
from imageio.v2 import imread

from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer
from utils_tool.utils import *
from pathlib import Path


class Change_Perception:
    def __init__(self, args):
        # Configuration
        self.args = args

        SRC_DIR = Path(__file__).resolve().parent
        def resolve_here(p) -> Path:
            p = Path(p)
            return p if p.is_absolute() else (SRC_DIR / p)

        self.mean = [0.39073 * 255, 0.38623 * 255, 0.32989 * 255]
        self.std = [0.15329 * 255, 0.14628 * 255, 0.13648 * 255]

        print(f"args: {args}")

        # Load vocabulary
        # ---- build paths ----
        list_path_dir = resolve_here(self.args.list_path)  # e.g. src/data/LEVIR_MCI
        vocab_path = list_path_dir / f"{self.args.vocab_file}.json"
        checkpoint_path = resolve_here(self.args.checkpoint)

        if not vocab_path.is_file():
            # running from root
            vocab_path = Path("Multi_change") / Path(self.args.list_path) / f"{self.args.vocab_file}.json"
            checkpoint_path = Path("Multi_change") / Path(checkpoint_path)

        # now load
        with open(vocab_path, "r") as f:
            self.word_vocab = json.load(f)

        # Load model checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Initialize models
        self.encoder = Encoder(self.args.network)
        self.encoder_trans = AttentiveEncoder(
            train_stage=None,
            n_layers=self.args.n_layers,
            feature_size=[self.args.feat_size, self.args.feat_size, self.args.encoder_dim],
            heads=self.args.n_heads,
            dropout=self.args.dropout
        )
        self.decoder = DecoderTransformer(
            encoder_dim=self.args.encoder_dim,
            feature_dim=self.args.feature_dim,
            vocab_size=len(self.word_vocab),
            max_lengths=self.args.max_length,
            word_vocab=self.word_vocab,
            n_head=self.args.n_heads,
            n_layers=self.args.decoder_n_layers,
            dropout=self.args.dropout
        )

        # Load weights
        self.encoder.load_state_dict(checkpoint["encoder_dict"])
        self.encoder_trans.load_state_dict(checkpoint["encoder_trans_dict"], strict=False)
        self.decoder.load_state_dict(checkpoint["decoder_dict"])

        # Move to GPU if available
        self.device = torch.device(f"cuda:{self.args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.encoder.to(self.device).eval()
        self.encoder_trans.to(self.device).eval()
        self.decoder.to(self.device).eval()

    def preprocess(self, path_A, path_B):
        imgA = imread(path_A).astype(np.float32)
        imgB = imread(path_B).astype(np.float32)

        # Channel-first
        imgA = imgA.transpose(2, 0, 1)
        imgB = imgB.transpose(2, 0, 1)

        # Normalize
        for i in range(3):
            imgA[i] = (imgA[i] - self.mean[i]) / self.std[i]
            imgB[i] = (imgB[i] - self.mean[i]) / self.std[i]

        # Resize
        if imgA.shape[1] != 256 or imgA.shape[2] != 256:
            imgA = cv2.resize(imgA, (256, 256))
            imgB = cv2.resize(imgB, (256, 256))

        # To tensor
        imgA = torch.from_numpy(imgA).float().unsqueeze(0).to(self.device)
        imgB = torch.from_numpy(imgB).float().unsqueeze(0).to(self.device)

        return imgA, imgB

    def generate_change_caption(self, path_A, path_B):
        imgA, imgB = self.preprocess(path_A, path_B)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, _ = self.encoder_trans(feat1, feat2)
        seq = self.decoder.sample(feat1, feat2, k=1)
        # Filter out special tokens
        filtered = [w for w in seq if w not in {self.word_vocab["<START>"], self.word_vocab["<END>"], self.word_vocab["<NULL>"]}]
        caption = " ".join([list(self.word_vocab.keys())[i] for i in filtered])
        print("Caption:", caption)
        return caption

    def change_detection(self, path_A, path_B, savepath):
        imgA, imgB = self.preprocess(path_A, path_B)
        feat1, feat2 = self.encoder(imgA, imgB)
        feat1, feat2, seg_pre = self.encoder_trans(feat1, feat2)

        # Segmentation output
        pred_seg = seg_pre.detach().cpu().numpy()
        pred = np.argmax(pred_seg, axis=1)[0].astype(np.uint8)

        # Color map
        pred_rgb = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
        pred_rgb[pred == 1] = [0, 255, 255]
        pred_rgb[pred == 2] = [0, 0, 255]

        # Save
        cv2.imwrite(savepath, pred_rgb)
        print("Mask saved at", savepath)
        return pred

    def compute_object_num(self, mask, object_type):
        mask_cp = np.zeros_like(mask, dtype=np.uint8)
        if object_type == "road":
            mask_cp[mask == 1] = 255
        elif object_type == "building":
            mask_cp[mask == 2] = 255
        labels = measure.label(mask_cp, connectivity=2)
        props = measure.regionprops(labels)
        # Filter small regions
        bboxes = [prop.bbox for prop in props if prop.area > 5]
        count = len(bboxes)
        print(f"Found {count} {object_type}(s)")
        return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remote_Sensing_Image_Change_Interpretation")
    # Model & data args
    parser.add_argument("--data_folder", default="data", help="Folder with data files")
    parser.add_argument("--list_path", default="data/LEVIR_MCI", help="Path to lists and vocab")
    parser.add_argument("--vocab_file", default="vocab", help="Vocabulary filename (without .json)")
    parser.add_argument("--max_length", type=int, default=41, help="Max sequence length")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID")
    parser.add_argument("--checkpoint", default="models_ckpt/MCI_model.pth", help="Checkpoint path")
    parser.add_argument("--network", default="segformer-mit_b1", help="Backbone network")
    parser.add_argument("--encoder_dim", type=int, default=512, help="Encoder feature dim")
    parser.add_argument("--feat_size", type=int, default=16, help="Encoder feature map size")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n_layers", type=int, default=3, help="Layers in attentive encoder")
    parser.add_argument("--decoder_n_layers", type=int, default=1, help="Decoder layers")
    parser.add_argument("--feature_dim", type=int, default=512, help="Decoder feature dim")
    # Inference args
    parser.add_argument("--imgA_path", required=True, help="Path to image A")
    parser.add_argument("--imgB_path", required=True, help="Path to image B")
    parser.add_argument("--mask_save_path", required=True, help="Where to save the mask")

    args = parser.parse_args()

    cp = Change_Perception(args)
    cp.generate_change_caption(args.imgA_path, args.imgB_path)
    cp.change_detection(args.imgA_path, args.imgB_path, args.mask_save_path)
