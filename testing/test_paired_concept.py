import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from toolkit.config_modules import DatasetConfig
from toolkit.data_loader import get_dataloader_from_datasets


class FakeSD:
    encode_control_in_text_embeddings = False
    encode_first_frame_in_text_embeddings = False
    use_raw_control_images = False
    dopsd_self_ref = False
    te_padding_side = "right"
    vae = SimpleNamespace(config=SimpleNamespace())
    unet = SimpleNamespace(config=SimpleNamespace())

    def get_bucket_divisibility(self):
        return 8

    def get_latent_space_version(self):
        return "test"

    def get_text_embedding_space_version(self):
        return "test"


class PairedConceptTest(unittest.TestCase):
    def test_loader_keeps_images_and_captions_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            positive = root / "positive.png"
            negative = root / "negative.png"
            Image.new("RGB", (64, 64), "white").save(positive)
            Image.new("RGB", (64, 64), "black").save(negative)
            manifest = root / "pairs.json"
            manifest.write_text(json.dumps({str(positive): {
                "caption": "a woman with bigbutt",
                "negative_image": str(negative),
                "negative_caption": "a woman",
            }}))
            config = DatasetConfig(
                type="paired_image", dataset_path=str(manifest), trigger_word="bigbutt",
                resolution=32, num_workers=0, buckets=True,
            )
            batch = next(iter(get_dataloader_from_datasets([config], batch_size=1, sd=FakeSD())))
            self.assertEqual(batch.get_caption_list(), ["a woman with bigbutt"])
            self.assertEqual(batch.paired_negative_batch.get_caption_list(), ["a woman"])
            self.assertEqual(batch.tensor.shape, batch.paired_negative_batch.tensor.shape)
            self.assertEqual(batch.file_items[0].path, str(positive))
            self.assertEqual(batch.paired_negative_batch.file_items[0].path, str(negative))

    def test_difference_loss_has_gradients_and_favors_noisy_timesteps(self):
        positive = torch.tensor([[[[1.0]]], [[[1.0]]]], requires_grad=True)
        negative = torch.zeros_like(positive, requires_grad=True)
        target_positive = torch.zeros_like(positive)
        target_negative = torch.zeros_like(negative)
        losses = SDTrainer.calculate_paired_concept_losses(
            positive, negative, torch.zeros_like(negative),
            target_positive, target_negative, torch.tensor([100.0, 900.0]),
            1000, 2,
        )
        self.assertAlmostEqual(losses[2].item(), (0.01 + 0.81) / 2)
        sum(losses).backward()
        self.assertIsNotNone(positive.grad)
        self.assertIsNotNone(negative.grad)
        self.assertNotEqual(negative.grad.abs().sum().item(), 0)

    def test_training_step_backpropagates_all_three_losses(self):
        class Batch:
            def __init__(self, latent, caption):
                self.latents = latent
                self.tensor = None
                self.prompt_embeds = torch.zeros(1, 1)
                self.caption = caption

            def get_caption_list(self):
                return [self.caption]

            def get_network_weight_list(self):
                return [1.0]

        class Network:
            multiplier = [1.0]

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

        trainer = SDTrainer.__new__(SDTrainer)
        trainer.device_torch = torch.device("cpu")
        trainer.train_config = SimpleNamespace(
            dtype="float32", num_train_timesteps=1000,
            paired_difference_timestep_power=2.0,
            paired_difference_multiplier=1.0,
            paired_preservation_multiplier=1.0,
        )
        trainer.sd = SimpleNamespace(add_noise=lambda latent, noise, timestep: latent + noise)
        trainer.network = Network()
        trainer.accelerator = SimpleNamespace(backward=lambda loss: loss.backward())
        trainer.additional_logs = {}
        trainer.preprocess_batch = lambda batch: batch
        noise = torch.ones(1, 1, 1, 1)
        timestep = torch.tensor([900.0])
        trainer.process_general_training_batch = lambda batch: (
            batch.latents + noise, noise, timestep, batch.get_caption_list(), None
        )
        weight = torch.nn.Parameter(torch.tensor(0.2))
        trainer.get_prior_prediction = lambda **kwargs: torch.zeros_like(kwargs["noisy_latents"])
        trainer.predict_noise = lambda **kwargs: kwargs["noisy_latents"] * weight
        positive = Batch(torch.ones(1, 1, 1, 1), "bigbutt woman")
        positive.paired_negative_batch = Batch(torch.zeros(1, 1, 1, 1), "woman")

        loss = trainer.train_paired_concept_accumulation(positive)
        self.assertTrue(torch.isfinite(loss))
        self.assertNotEqual(weight.grad.item(), 0)
        self.assertEqual(set(trainer.additional_logs), {
            "loss/paired_positive", "loss/paired_preservation", "loss/paired_difference"
        })


if __name__ == "__main__":
    unittest.main()
