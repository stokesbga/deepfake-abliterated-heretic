# Third-party notices

This project integrates DreamID-V Faster and Wan 2.1 components. It does not claim
authorship of their model architecture, pretrained weights, or research results.

## Vendored code

Source: https://github.com/bytedance/DreamID-V

Pinned revision: `9b589940577559c91481fb3a13bae000a55f97a1`.

Files under `src/deepfakeheretic/_vendor/wan/` (except the local attention adapter)
and `src/deepfakeheretic/_vendor/dwpose/` were copied from that revision. Original
copyright notices are retained. The upstream Apache 2.0 license is reproduced at
`src/deepfakeheretic/_vendor/LICENSE.txt`.

Upstream attributes the Wan model/VAE to the Alibaba Wan Team, DreamID-V to
ByteDance and its collaborators, DWPose helpers to Alibaba and DWPose contributors,
and the UniPC scheduler to Hugging Face Diffusers with Wan flow-matching changes.
The origin manifest records paths and checksums. Package initializers and the
SDPA attention adapter were added locally.

## Model assets

`src/deepfakeheretic/assets.json` records exact source revisions, byte sizes, and
SHA-256 hashes for every downloaded checkpoint. Downloads remain separate from
the Python source distribution.

- DreamID-V Faster, DWPose detector/pose files:
  https://huggingface.co/XuGuo699/DreamID-V
- Wan VAE: https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B
- Fixed conditioning context: pinned DreamID-V GitHub revision above.

Refer to the original repositories and model cards for their terms, attribution,
intended use, and limitations. The source-code license does not replace model or
dataset terms.

