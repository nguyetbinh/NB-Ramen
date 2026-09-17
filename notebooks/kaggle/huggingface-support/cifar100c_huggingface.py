"""Pinned Hugging Face transport with independent original NumPy checksums.

The MD5 table is published by TorchUncertainty's CIFAR100C implementation:
https://torch-uncertainty.github.io/_modules/torch_uncertainty/datasets/classification/cifar/cifar_c.html
It identifies the original .npy bytes, including image order and NPY headers.
This acquisition does not claim to have downloaded or hashed the Zenodo tar.
"""

from pathlib import Path

HF_REPOSITORY = "WNJXYK/TTA-CIFAR-100-C"
HF_REVISION = "a12f0bcc1da33fa26d8c76ce8c1fb32e6f913bea"
CIFAR100C_NPY_MD5 = {
    "brightness.npy": "f22d7195aecd6abb541e27fca230c171",
    "contrast.npy": "322bb385f1d05154ee197ca16535f71e",
    "defocus_blur.npy": "d923e3d9c585a27f0956e2f2ad832564",
    "elastic_transform.npy": "a0792bd6581f6810878be71acedfc65a",
    "fog.npy": "4efc7ebd5e82b028bdbe13048e3ea564",
    "frost.npy": "3a39c6823bdfaa0bf8b12fe7004b8117",
    "gaussian_blur.npy": "5204ba0d557839772ef5a4196a052c3e",
    "gaussian_noise.npy": "ecc4d366eac432bdf25c024086f5e97d",
    "glass_blur.npy": "0bf384f38e5ccbf8dd479d9059b913e1",
    "impulse_noise.npy": "3b3c210ddfa0b5cb918ff4537a429fef",
    "jpeg_compression.npy": "c851b7f1324e1d2ffddeb76920576d11",
    "labels.npy": "bb4026e9ce52996b95f439544568cdb2",
    "motion_blur.npy": "732a7e2e54152ff97c742d4c388c5516",
    "pixelate.npy": "96c00c60f144539e14cffb02ddbd0640",
    "saturate.npy": "c0697e9fdd646916a61e9c312c77bf6b",
    "shot_noise.npy": "b0a1fa6e1e465a747c1b204b1914048a",
    "snow.npy": "0237be164583af146b7b144e73b43465",
    "spatter.npy": "12ccf41d62564d36e1f6a6ada5022728",
    "speckle_noise.npy": "e3f215b1a0f9fd9fd6f0d1cf94a7ce99",
    "zoom_blur.npy": "0204613400c034a81c4830d5df81cb82",
}
CIFAR100C_HF_ACQUISITION = {
    "publisher": "Hugging Face community mirror",
    "repository": HF_REPOSITORY,
    "revision": HF_REVISION,
    "url": f"https://huggingface.co/datasets/{HF_REPOSITORY}/tree/{HF_REVISION}",
    "upstream_doi": "10.5281/zenodo.3555552",
    "verification": "reconstructed_original_npy_md5",
    "files": dict(CIFAR100C_NPY_MD5),
}


def verify_huggingface_cifar100c_files(root):
    """Check all twenty original files, not just mirror sizes or self-reported hashes."""
    try:
        from .artifact_provenance import ProvenanceError, _walk_regular_files, checksum_regular_file
    except ImportError:
        from artifact_provenance import ProvenanceError, _walk_regular_files, checksum_regular_file

    files = dict(_walk_regular_files(Path(root)))
    if set(files) != set(CIFAR100C_NPY_MD5):
        raise ProvenanceError("Hugging Face CIFAR-100-C requires exactly the twenty original NPY files")
    actual = {}
    for name, expected in CIFAR100C_NPY_MD5.items():
        digest = checksum_regular_file(files[name], "md5")["checksum"]
        if digest != expected:
            raise ProvenanceError(f"Original CIFAR-100-C NPY checksum mismatch: {name}")
        actual[name] = digest
    return {**CIFAR100C_HF_ACQUISITION, "files": actual}
