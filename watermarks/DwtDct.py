import numpy as np
import cv2
from PIL import Image
import pywt


class DwtDCT:
    def __init__(self, use_svd=False):
        self._scales = [0, 36, 36]
        self._block = 4
        self._use_svd = use_svd

    def encode(self, images, watermarks):
        """
        Encodes watermarks into a batch of images.

        Args:
            images (list): List of Pillow images to encode.
            watermarks (numpy.ndarray): Batch of watermarks with shape (n, 32).

        Returns:
            list: List of encoded Pillow images.
        """
        batch_size = len(images)
        if batch_size != watermarks.shape[0]:
            raise ValueError("Number of images and watermarks must match in the batch.")

        encoded_images = []
        for idx, image in enumerate(images):
            watermark = watermarks[idx]
            bgr = np.array(image)
            encoded_bgr = self._encode_single(bgr, watermark)
            encoded_image = Image.fromarray(encoded_bgr)
            encoded_images.append(encoded_image)
        return encoded_images

    def decode(self, images):
        """
        Decodes watermarks from a batch of images.

        Args:
            images (list): List of Pillow images to decode.

        Returns:
            numpy.ndarray: Decoded watermarks with shape (n, 32).
        """
        decoded_watermarks = []
        for image in images:
            bgr = np.array(image)
            bits = self._decode_single(bgr)
            decoded_watermarks.append(bits)
        return np.array(decoded_watermarks)

    def _encode_single(self, bgr, watermark):
        """
        Encodes a single image with a single watermark.

        Args:
            bgr (numpy.ndarray): BGR image as a numpy array.
            watermark (numpy.ndarray): 1D array of watermark bits with length 32.

        Returns:
            numpy.ndarray: Encoded BGR image.
        """
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:row // 4 * 4, :col // 4 * 4, channel], 'haar')
            self.encode_frame(ca1, watermark, self._scales[channel])

            yuv[:row // 4 * 4, :col // 4 * 4, channel] = pywt.idwt2((ca1, (v1, h1, d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def _decode_single(self, bgr):
        """
        Decodes a single image to extract the watermark.

        Args:
            bgr (numpy.ndarray): BGR image as a numpy array.

        Returns:
            numpy.ndarray: Decoded watermark bits as a 1D array of length 32.
        """
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for _ in range(32)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:row // 4 * 4, :col // 4 * 4, channel], 'haar')
            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))
        bits = (np.array(avgScores) * 255 > 127).astype(int)
        return bits

    def encode_frame(self, frame, watermark, scale):
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block: i * self._block + self._block,
                              j * self._block: j * self._block + self._block]
                wmBit = watermark[num % 32]

                if self._use_svd:
                    diffusedBlock = self.diffuse_dct_svd(block, wmBit, scale)
                else:
                    diffusedBlock = self.diffuse_dct_matrix(block, wmBit, scale)

                frame[i * self._block: i * self._block + self._block,
                      j * self._block: j * self._block + self._block] = diffusedBlock

                num += 1

    def decode_frame(self, frame, scale, scores):
        (row, col) = frame.shape
        num = 0

        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block: i * self._block + self._block,
                              j * self._block: j * self._block + self._block]

                if self._use_svd:
                    score = self.infer_dct_svd(block, scale)
                else:
                    score = self.infer_dct_matrix(block, scale)

                wmBit = num % 32
                scores[wmBit].append(score)
                num += 1

        return scores

    def diffuse_dct_matrix(self, block, wmBit, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block
        val = block[i][j]
        if val >= 0.0:
            block[i][j] = (val // scale + 0.25 + 0.5 * wmBit) * scale
        else:
            val = abs(val)
            block[i][j] = -1.0 * (val // scale + 0.25 + 0.5 * wmBit) * scale
        return block

    def infer_dct_matrix(self, block, scale):
        pos = np.argmax(abs(block.flatten()[1:])) + 1
        i, j = pos // self._block, pos % self._block

        val = block[i][j]
        if val < 0:
            val = abs(val)

        return int((val % scale) > 0.5 * scale)

    def diffuse_dct_svd(self, block, wmBit, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))
        s[0] = (s[0] // scale + 0.25 + 0.5 * wmBit) * scale
        return cv2.idct(np.dot(u, np.dot(np.diag(s), v)))

    def infer_dct_svd(self, block, scale):
        u, s, v = np.linalg.svd(cv2.dct(block))
        return int((s[0] % scale) > scale * 0.5)
