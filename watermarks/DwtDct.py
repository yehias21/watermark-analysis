import numpy as np
import cv2
import pywt

class WatermarkEmbedder:
    def __init__(self, watermarks=[], wmLen=8, scales=[0, 36, 36], block=4, use_svd=False):
        self._watermarks = watermarks
        self._wmLen = wmLen
        self._scales = scales
        self._block = block
        self._use_svd = use_svd

    def encode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:row // 4 * 4, :col // 4 * 4, channel], 'haar')
            self.encode_frame(ca1, self._scales[channel])

            yuv[:row // 4 * 4, :col // 4 * 4, channel] = pywt.idwt2((ca1, (v1, h1, d1)), 'haar')

        bgr_encoded = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return bgr_encoded

    def decode(self, bgr):
        (row, col, channels) = bgr.shape
        yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)

        scores = [[] for _ in range(self._wmLen)]
        for channel in range(2):
            if self._scales[channel] <= 0:
                continue

            ca1, (h1, v1, d1) = pywt.dwt2(yuv[:row // 4 * 4, :col // 4 * 4, channel], 'haar')
            scores = self.decode_frame(ca1, self._scales[channel], scores)

        avgScores = list(map(lambda l: np.array(l).mean(), scores))
        bits = (np.array(avgScores) * 255 > 127)
        return bits

    def encode_frame(self, frame, scale):
        (row, col) = frame.shape
        num = 0
        for i in range(row // self._block):
            for j in range(col // self._block):
                block = frame[i * self._block: i * self._block + self._block,
                              j * self._block: j * self._block + self._block]
                wmBit = self._watermarks[num % self._wmLen]

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

                wmBit = num % self._wmLen
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